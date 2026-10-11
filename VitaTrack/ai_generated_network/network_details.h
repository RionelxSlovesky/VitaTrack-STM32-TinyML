/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-10-10T21:01:03-0400
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[5]) {
   { .size_bytes = 6, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 6}}, .scale = {1, (const float[1]){0.026130691170692444}}, .zeropoint = {1, (const int16_t[1]){-41}}, .name = "serving_default_keras_tensor0_output" },
   { .size_bytes = 12, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 12}}, .scale = {1, (const float[1]){0.025825584307312965}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_0_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 8}}, .scale = {1, (const float[1]){0.02830592356622219}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_1_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){0.04178722947835922}}, .zeropoint = {1, (const int16_t[1]){-48}}, .name = "gemm_2_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_3_output" }
  },
  .nodes = (const stai_node_details[4]){
    {.id = 0, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* gemm_0 */
    {.id = 1, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* gemm_1 */
    {.id = 2, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* gemm_2 */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} } /* nl_3 */
  },
  .n_nodes = 4
};
#endif

