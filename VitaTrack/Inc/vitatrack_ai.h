#ifndef VITATRACK_AI_H
#define VITATRACK_AI_H

#include <stdint.h>

#define VITATRACK_AI_FEATURE_COUNT 6U

/*
 * Initialize the ST.AI runtime and generated network.
 * Returns 0 on success, negative on failure.
 */
int VitaTrack_AI_Init(void);

/*
 * Run one stress inference.
 *
 * features order:
 *   0 mean_rr_ms
 *   1 mean_hr_bpm
 *   2 sdnn_ms
 *   3 rmssd_ms
 *   4 pnn50_pct
 *   5 rr_range_ms
 *
 * probability receives the dequantized sigmoid output [0,1].
 * is_stress is set using VITATRACK_STRESS_THRESHOLD.
 *
 * Returns 0 on success, negative on failure.
 */
int VitaTrack_AI_Run(const float features[VITATRACK_AI_FEATURE_COUNT],
                     float *probability,
                     uint8_t *is_stress);

void VitaTrack_AI_Deinit(void);

#endif /* VITATRACK_AI_H */
