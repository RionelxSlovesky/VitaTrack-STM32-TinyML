/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "gpio.h"
#include "vitatrack_ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "ecg_replay.h"

#include "ecg_replay.h"
#include <math.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

static float stress_probability = 0.0f;
static uint8_t stress_detected = 0U;

typedef struct
{
    float mean_rr_ms;
    float mean_hr_bpm;
    float sdnn_ms;
    float rmssd_ms;
    float pnn50_pct;
    float rr_range_ms;

    uint16_t interval_count;

} HRVFeatures;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define ECG_BUFFER_SIZE 256
#define ECG_FS_HZ                250U
#define ECG_ADC_MIDPOINT         2048.0f
#define ECG_REFRACTORY_SAMPLES   63U   // ~252 ms at 250 Hz

#define VITATRACK_REPLAY_MODE 1
#define REPLAY_BLOCK_SIZE 128

#define RR_BUFFER_SIZE          256U
#define HRV_WINDOW_MS           60000.0f
#define HRV_MIN_INTERVALS       8U

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

uint16_t ecg_buffer[ECG_BUFFER_SIZE];

volatile uint8_t ecg_half_ready = 0;
volatile uint8_t ecg_full_ready = 0;

/* ECG filter state */
static float ecg_x_prev = 0.0f;
static float ecg_hp_prev = 0.0f;
static float ecg_lp_prev = 0.0f;

/* R-peak detector state */
static float ecg_filtered_prev = 0.0f;
static float ecg_energy_floor = 1.0f;
static float ecg_energy_peak = 1.0f;

static uint32_t ecg_sample_index = 0;
static uint32_t last_rpeak_index = 0;
static uint32_t rpeak_count = 0;

static float last_rr_ms = 0.0f;
static float heart_rate_bpm = 0.0f;

#if VITATRACK_REPLAY_MODE
static uint32_t replay_index = 0;
static uint8_t replay_finished = 0;
#endif

/* RR / HRV state */
static float rr_buffer[RR_BUFFER_SIZE];

static uint16_t rr_head = 0;
static uint16_t rr_count = 0;

static HRVFeatures hrv_features;
static volatile uint8_t hrv_features_ready = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

static void rr_buffer_add(float rr_ms);
static uint8_t compute_hrv_features(HRVFeatures *out);

static float ecg_filter_sample(uint16_t raw_sample);
static void ecg_detect_rpeak(float filtered_sample);
static void process_ecg_block(const uint16_t *buffer,
                              uint16_t length);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static float ecg_filter_sample(uint16_t raw_sample)
{
    /*
     * ADC is 12-bit, so the nominal midpoint is ~2048.
     * First remove the DC midpoint.
     */
    float x = (float)raw_sample - ECG_ADC_MIDPOINT;

    /*
     * First-order high-pass filter.
     * Approximate cutoff: 0.5 Hz at Fs = 250 Hz.
     *
     * y[n] = a * (y[n-1] + x[n] - x[n-1])
     */
    const float hp_alpha = 0.9876f;

    float hp = hp_alpha *
               (ecg_hp_prev + x - ecg_x_prev);

    ecg_x_prev = x;
    ecg_hp_prev = hp;

    /*
     * First-order low-pass filter.
     * Approximate cutoff: 35 Hz at Fs = 250 Hz.
     */
    const float lp_alpha = 0.4680f;

    ecg_lp_prev =
        ecg_lp_prev +
        lp_alpha * (hp - ecg_lp_prev);

    return ecg_lp_prev;
}

static void rr_buffer_add(float rr_ms)
{
    rr_buffer[rr_head] = rr_ms;

    rr_head++;

    if (rr_head >= RR_BUFFER_SIZE)
    {
        rr_head = 0;
    }

    if (rr_count < RR_BUFFER_SIZE)
    {
        rr_count++;
    }

    /*
     * Recalculate HRV whenever a new valid RR interval
     * arrives and a full analysis window is available.
     */
    if (compute_hrv_features(&hrv_features))
    {
        hrv_features_ready = 1;
    }
}

static uint8_t compute_hrv_features(HRVFeatures *out)
{
    if ((out == NULL) ||
        (rr_count < HRV_MIN_INTERVALS))
    {
        return 0;
    }

    /*
     * Walk backward through the circular buffer until
     * approximately 60 seconds of RR intervals are included.
     */
    float accumulated_ms = 0.0f;
    uint16_t selected = 0;

    while ((selected < rr_count) &&
           (accumulated_ms < HRV_WINDOW_MS))
    {
        uint16_t index =
            (uint16_t)(
                (rr_head +
                 RR_BUFFER_SIZE -
                 1U -
                 selected)
                % RR_BUFFER_SIZE
            );

        accumulated_ms += rr_buffer[index];
        selected++;
    }

    /*
     * Do not calculate ML features until we actually
     * have approximately one minute of ECG history.
     */
    if ((accumulated_ms < HRV_WINDOW_MS) ||
        (selected < HRV_MIN_INTERVALS))
    {
        return 0;
    }

    /*
     * Determine the oldest interval in the selected
     * rolling window.
     */
    uint16_t start =
        (uint16_t)(
            (rr_head +
             RR_BUFFER_SIZE -
             selected)
            % RR_BUFFER_SIZE
        );

    float sum = 0.0f;

    float rr_min =
        rr_buffer[start];

    float rr_max =
        rr_buffer[start];

    /*
     * Mean RR and range.
     */
    for (uint16_t i = 0;
         i < selected;
         i++)
    {
        uint16_t index =
            (uint16_t)(
                (start + i)
                % RR_BUFFER_SIZE
            );

        float rr =
            rr_buffer[index];

        sum += rr;

        if (rr < rr_min)
        {
            rr_min = rr;
        }

        if (rr > rr_max)
        {
            rr_max = rr;
        }
    }

    float mean_rr =
        sum / (float)selected;

    /*
     * SDNN, RMSSD and pNN50.
     */
    float variance_sum = 0.0f;
    float diff_square_sum = 0.0f;

    uint16_t nn50_count = 0;

    float previous_rr = 0.0f;

    for (uint16_t i = 0;
         i < selected;
         i++)
    {
        uint16_t index =
            (uint16_t)(
                (start + i)
                % RR_BUFFER_SIZE
            );

        float rr =
            rr_buffer[index];

        float deviation =
            rr - mean_rr;

        variance_sum +=
            deviation * deviation;

        if (i > 0U)
        {
            float difference =
                rr - previous_rr;

            diff_square_sum +=
                difference * difference;

            if (fabsf(difference) > 50.0f)
            {
                nn50_count++;
            }
        }

        previous_rr = rr;
    }

    out->mean_rr_ms =
        mean_rr;

    out->mean_hr_bpm =
        60000.0f / mean_rr;

    out->sdnn_ms =
        sqrtf(
            variance_sum /
            (float)(selected - 1U)
        );

    out->rmssd_ms =
        sqrtf(
            diff_square_sum /
            (float)(selected - 1U)
        );

    out->pnn50_pct =
        100.0f *
        (float)nn50_count /
        (float)(selected - 1U);

    out->rr_range_ms =
        rr_max - rr_min;

    out->interval_count =
        selected;

    return 1;
}

static void ecg_detect_rpeak(float filtered_sample)
{
    /*
     * Lightweight QRS detector:
     *
     * filtered ECG
     *      ↓
     * derivative
     *      ↓
     * square / energy
     *      ↓
     * adaptive threshold
     *      ↓
     * refractory-period check
     */

    float derivative =
        filtered_sample - ecg_filtered_prev;

    ecg_filtered_prev = filtered_sample;

    float energy = derivative * derivative;

    /*
     * Slowly varying estimate of normal/background energy.
     */
    ecg_energy_floor =
        0.995f * ecg_energy_floor +
        0.005f * energy;

    /*
     * Track recent high-energy activity.
     */
    if (energy > ecg_energy_peak)
    {
        ecg_energy_peak = energy;
    }
    else
    {
        ecg_energy_peak =
            0.999f * ecg_energy_peak +
            0.001f * ecg_energy_floor;
    }

    /*
     * Adaptive threshold between baseline and recent peaks.
     */
    float threshold =
        ecg_energy_floor +
        0.30f *
        (ecg_energy_peak - ecg_energy_floor);

    /*
     * Ignore the first two seconds while filters and
     * threshold estimates stabilize.
     */
    if (ecg_sample_index > (2U * ECG_FS_HZ))
    {
        uint32_t samples_since_peak =
            ecg_sample_index - last_rpeak_index;

        if ((energy > threshold) &&
            ((rpeak_count == 0U) ||
             (samples_since_peak >
              ECG_REFRACTORY_SAMPLES)))
        {
            if (rpeak_count > 0U)
            {
                float rr_ms =
                    (1000.0f *
                     (float)samples_since_peak) /
                    (float)ECG_FS_HZ;

                /*
                 * Reject obviously implausible intervals
                 * in this prototype.
                 */
                if ((rr_ms >= 300.0f) &&
                    (rr_ms <= 2000.0f))
                {
                    last_rr_ms = rr_ms;

                    heart_rate_bpm =
                        60000.0f / last_rr_ms;

                    rr_buffer_add(rr_ms);
                }
            }

            last_rpeak_index = ecg_sample_index;
            rpeak_count++;
        }
    }

    ecg_sample_index++;
}


static void process_ecg_block(const uint16_t *buffer,
                              uint16_t length)
{
    for (uint16_t i = 0; i < length; i++)
    {
        float filtered =
            ecg_filter_sample(buffer[i]);

        ecg_detect_rpeak(filtered);
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */

  #if !VITATRACK_REPLAY_MODE

  if (HAL_ADC_Start_DMA(&hadc1,
						  (uint32_t *)ecg_buffer,
						  ECG_BUFFER_SIZE) != HAL_OK)
  {
	  Error_Handler();
  }

  if (HAL_TIM_Base_Start(&htim2) != HAL_OK)
  {
	Error_Handler();
  }

  if (VitaTrack_AI_Init() != 0)
  {
      Error_Handler();
  }

  #endif

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

  #if VITATRACK_REPLAY_MODE

      if (!replay_finished)
      {
          uint32_t remaining =
              ECG_REPLAY_LENGTH - replay_index;

          uint16_t block_length =
              (remaining > REPLAY_BLOCK_SIZE)
              ? REPLAY_BLOCK_SIZE
              : (uint16_t)remaining;

          process_ecg_block(
              &ecg_replay_data[replay_index],
              block_length
          );

          replay_index += block_length;

          if (replay_index >= ECG_REPLAY_LENGTH)
          {
              replay_finished = 1;
          }
      }

  #else

      if (ecg_half_ready)
      {
          ecg_half_ready = 0;

          process_ecg_block(
              &ecg_buffer[0],
              ECG_BUFFER_SIZE / 2
          );
      }

      if (ecg_full_ready)
      {
          ecg_full_ready = 0;

          process_ecg_block(
              &ecg_buffer[ECG_BUFFER_SIZE / 2],
              ECG_BUFFER_SIZE / 2
          );
      }

  #endif

      /* Run TinyML inference whenever a new HRV window is ready */
      if (hrv_features_ready)
      {
          hrv_features_ready = 0;

          float features[VITATRACK_AI_FEATURE_COUNT] =
          {
              hrv_features.mean_rr_ms,
              hrv_features.mean_hr_bpm,
              hrv_features.sdnn_ms,
              hrv_features.rmssd_ms,
              hrv_features.pnn50_pct,
              hrv_features.rr_range_ms
          };

          if (VitaTrack_AI_Run(
                  features,
                  &stress_probability,
                  &stress_detected) != 0)
          {
              Error_Handler();
          }
      }
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc->Instance == ADC1)
    {
        ecg_half_ready = 1;
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc->Instance == ADC1)
    {
        ecg_full_ready = 1;
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
