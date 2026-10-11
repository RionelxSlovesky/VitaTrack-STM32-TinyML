#include "vitatrack_ai.h"

#include <math.h>
#include <stddef.h>

#include "network.h"
#include "user_init.h"
#include "stai.h"
#include "vitatrack_stress_model_metadata.h"

STAI_NETWORK_CONTEXT_DECLARE(vitatrack_network_context,
                             STAI_NETWORK_CONTEXT_SIZE);

STAI_ALIGNED(32)
static uint8_t vitatrack_activations[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];

static stai_ptr vitatrack_activation_ptrs[STAI_NETWORK_ACTIVATIONS_NUM] = {
    vitatrack_activations
};

static stai_ptr vitatrack_inputs[STAI_NETWORK_IN_NUM];
static stai_ptr vitatrack_outputs[STAI_NETWORK_OUT_NUM];

static stai_size vitatrack_input_count = 0;
static stai_size vitatrack_output_count = 0;
static uint8_t vitatrack_ai_initialized = 0U;

static int8_t quantize_s8(float value, float scale, int32_t zero_point)
{
    float qf = (value / scale) + (float)zero_point;
    long q = lroundf(qf);

    if (q > 127L)
    {
        q = 127L;
    }
    else if (q < -128L)
    {
        q = -128L;
    }

    return (int8_t)q;
}

int VitaTrack_AI_Init(void)
{
    stai_return_code rc;

    if (vitatrack_ai_initialized)
    {
        return 0;
    }

    rc = stai_runtime_init();
    if (rc != STAI_SUCCESS)
    {
        return -1;
    }

    rc = user_stai_network_init(vitatrack_network_context);
    if (rc != STAI_SUCCESS)
    {
        (void)stai_runtime_deinit();
        return -2;
    }

    rc = stai_network_set_activations(
        vitatrack_network_context,
        vitatrack_activation_ptrs,
        STAI_NETWORK_ACTIVATIONS_NUM
    );
    if (rc != STAI_SUCCESS)
    {
        (void)stai_network_deinit(vitatrack_network_context);
        (void)stai_runtime_deinit();
        return -3;
    }

    rc = stai_network_get_inputs(
        vitatrack_network_context,
        vitatrack_inputs,
        &vitatrack_input_count
    );
    if ((rc != STAI_SUCCESS) ||
        (vitatrack_input_count != STAI_NETWORK_IN_NUM))
    {
        (void)stai_network_deinit(vitatrack_network_context);
        (void)stai_runtime_deinit();
        return -4;
    }

    rc = stai_network_get_outputs(
        vitatrack_network_context,
        vitatrack_outputs,
        &vitatrack_output_count
    );
    if ((rc != STAI_SUCCESS) ||
        (vitatrack_output_count != STAI_NETWORK_OUT_NUM))
    {
        (void)stai_network_deinit(vitatrack_network_context);
        (void)stai_runtime_deinit();
        return -5;
    }

    vitatrack_ai_initialized = 1U;
    return 0;
}

int VitaTrack_AI_Run(const float features[VITATRACK_AI_FEATURE_COUNT],
                     float *probability,
                     uint8_t *is_stress)
{
    if ((!vitatrack_ai_initialized) ||
        (features == NULL) ||
        (probability == NULL) ||
        (is_stress == NULL))
    {
        return -1;
    }

    int8_t *input = (int8_t *)vitatrack_inputs[0];

    for (uint32_t i = 0; i < VITATRACK_AI_FEATURE_COUNT; i++)
    {
        float normalized =
            (features[i] - vitatrack_feature_mean[i]) /
            vitatrack_feature_std[i];

        input[i] = quantize_s8(
            normalized,
            STAI_NETWORK_IN_1_SCALE,
            STAI_NETWORK_IN_1_ZERO_POINT
        );
    }

    stai_return_code rc =
        stai_network_run(vitatrack_network_context, STAI_MODE_SYNC);

    if (rc != STAI_SUCCESS)
    {
        return -2;
    }

    int8_t q_output = ((int8_t *)vitatrack_outputs[0])[0];

    float p =
        ((float)q_output -
         (float)STAI_NETWORK_OUT_1_ZERO_POINT) *
        STAI_NETWORK_OUT_1_SCALE;

    if (p < 0.0f)
    {
        p = 0.0f;
    }
    else if (p > 1.0f)
    {
        p = 1.0f;
    }

    *probability = p;
    *is_stress =
        (p >= VITATRACK_STRESS_THRESHOLD) ? 1U : 0U;

    return 0;
}

void VitaTrack_AI_Deinit(void)
{
    if (vitatrack_ai_initialized)
    {
        (void)stai_network_deinit(vitatrack_network_context);
        (void)stai_runtime_deinit();
        vitatrack_ai_initialized = 0U;
    }
}
