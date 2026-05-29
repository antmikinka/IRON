// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reduction kernel for AIE2 (NPU)
// Supports: sum, max, min along the reduction dimension (mean is AIE2P-only)
//
// 600s hang modeling fix (this agent, feature/operator-reduction):
// Paired with design.py L3 staging + 4D TAPs + chunk-first depth (ref
// /tmp/reduction_hw_long.log + conv3d a2d5243/4c15030 + conv2d agent).
// No logic change here; extern "C" closure retained for aiecc symbol res.
//
// AUDITOR FIX (AIE2 / AIE2P Kernel Vectorization & Accumulator Discipline):
// - Fixed erroneous vector<bfloat16,N> accumulator in reduction_sum_bf16_vector (was
//   using aie::add + reduce_add directly on bf16 vector, violating AccumOrOp concept
//   for mac/reduce_add paths; same class of bug fixed on conv2d/conv3d).
// - Now uses aie::accum<accfloat,16> + mac idiom + to_vector<float> + reduce_add on float
//   (exact pattern from post-fix conv* aie2/aie2p kernels that enabled 600s NPU runs).
// - event0()/event1() remain only on hot vectorized path (scalars untouched).
// - extern "C" and reduction_*_bf16_vector signatures match Kernel() decls in
//   iron/operators/reduction/design.py exactly.
// References: 600s logs (/tmp/reduction_hw_long.log etc from iron314 runs on
// feature/operator-reduction), design resource agents (SPEC-011, design.py my_reduction),
// conv3d/conv2d auditor fixes in their worktrees.
//
// Cross-audit with aie2p/reduction.cc (which already used accum for sum/mean).

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

extern "C" {

/**
 * Reduction Sum Kernel - AIE2 optimized
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (sum of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_sum_bf16_scalar(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    bfloat16 acc = bfloat16(0.0f);

    for (int i = 0; i < reduction_size; i++) {
        acc += input[i];
    }

    output[0] = acc;
}

/**
 * Reduction Sum Kernel - Vectorized version for AIE2
 * Uses vector load and proper accum<accfloat> discipline (AUDITOR FIX)
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (sum of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_sum_bf16_vector(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    constexpr int vec_factor = 16; // Standardized to 16 for AIE2 vector/accum compatibility (matches aie2p/conv patterns)

    event0();

    bfloat16 *__restrict pIn = input;
    bfloat16 *__restrict pOut = output;

    // Initialize accumulator using from_vector + accfloat (AUDITOR: fixes AccumOrOp failures on add/reduce_add)
    aie::accum<accfloat, vec_factor> acc_vec;
    acc_vec.from_vector(aie::zeros<float, vec_factor>(), 0);

    const int F = reduction_size / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<bfloat16, vec_factor> in_vec = aie::load_v<vec_factor>(pIn);
        pIn += vec_factor;
        // Use mac with ones vector for sum (mulacc-by-1 idiom) - addresses bf16 accumulation compatibility
        auto ones = aie::broadcast<bfloat16, vec_factor>(bfloat16(1.0f));
        acc_vec = aie::mac(acc_vec, in_vec, ones);
    }

    // Horizontal sum using reduce_add on float vector (standard post-fix pattern)
    aie::vector<float, vec_factor> red = acc_vec.to_vector<float>();
    float sum_f = aie::reduce_add(red);

    // Handle remaining elements (accumulate in float for precision)
    const int remainder = reduction_size % vec_factor;
    for (int i = 0; i < remainder; i++) {
        sum_f += static_cast<float>(pIn[i]);
    }

    pOut[0] = static_cast<bfloat16>(sum_f);

    event1();
}

/**
 * Reduction Max Kernel - AIE2 optimized
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (max of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_max_bf16_scalar(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    bfloat16 max_val = input[0];

    for (int i = 1; i < reduction_size; i++) {
        max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    output[0] = max_val;
}

/**
 * Reduction Max Kernel - Vectorized version for AIE2
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (max of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_max_bf16_vector(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    constexpr int vec_factor = 16;

    event0();

    bfloat16 *__restrict pIn = input;
    bfloat16 *__restrict pOut = output;

    // Vectorized max using AIE native ops (no scalar inner loop for fast clean compile on AIE2/AIE2P)
    aie::vector<bfloat16, vec_factor> max_v = aie::broadcast<bfloat16, vec_factor>(bfloat16(-3.4028235e+38f));

    const int F = reduction_size / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<bfloat16, vec_factor> in_vec = aie::load_v<vec_factor>(pIn);
        pIn += vec_factor;

        max_v = aie::max(max_v, in_vec);
    }

    bfloat16 result = aie::reduce_max(max_v);

    // Handle remaining elements
    const int remainder = reduction_size % vec_factor;
    for (int i = 0; i < remainder; i++) {
        if (pIn[i] > result)
            result = pIn[i];
    }

    pOut[0] = result;

    event1();
}

/**
 * Reduction Min Kernel - AIE2 optimized
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (min of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_min_bf16_scalar(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    bfloat16 min_val = input[0];

    for (int i = 1; i < reduction_size; i++) {
        min_val = (input[i] < min_val) ? input[i] : min_val;
    }

    output[0] = min_val;
}

/**
 * Reduction Min Kernel - Vectorized version for AIE2
 *
 * @param input - Input tensor [reduction_dim]
 * @param output - Output scalar (min of all elements)
 * @param reduction_size - Size of the reduction dimension
 */
void reduction_min_bf16_vector(bfloat16 *input, bfloat16 *output, int reduction_size)
{
    constexpr int vec_factor = 16;

    event0();

    bfloat16 *__restrict pIn = input;
    bfloat16 *__restrict pOut = output;

    // Vectorized min using AIE native ops (no scalar inner loop for fast clean compile on AIE2/AIE2P)
    aie::vector<bfloat16, vec_factor> min_v = aie::broadcast<bfloat16, vec_factor>(bfloat16(3.4028235e+38f));

    const int F = reduction_size / vec_factor;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<bfloat16, vec_factor> in_vec = aie::load_v<vec_factor>(pIn);
        pIn += vec_factor;

        min_v = aie::min(min_v, in_vec);
    }

    bfloat16 result = aie::reduce_min(min_v);

    // Handle remaining elements
    const int remainder = reduction_size % vec_factor;
    for (int i = 0; i < remainder; i++) {
        if (pIn[i] < result)
            result = pIn[i];
    }

    pOut[0] = result;

    event1();
}

} // end extern "C" for C-linkage kernels (fix for symbol resolution in aiecc link, matching reduction.cc fix)
