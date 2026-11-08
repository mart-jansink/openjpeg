/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2026, Mart Jansink
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OPJ_DC_SHIFT_H
#define OPJ_DC_SHIFT_H

/**
@file dc_shift.h
@brief Row kernels for the decoder's DC level shift (opj_tcd_dc_level_shift_decode)

Each kernel shifts and clamps one row of n samples in place:
- opj_dc_shift_row_int: 5/3 (reversible) planes,
  x = clamp(x + shift, min, max), with 32-bit (wrapping) addition;
- opj_dc_shift_row_float: 9/7 (irreversible) planes holding float bit
  patterns, x = clamp(lrintf(f) + shift, min, max), where f above
  (float)INT_MAX gives max and f below INT_MIN gives min.

The NEON versions give exactly the same results as the scalar ones:
vcvtnq_s32_f32 rounds to nearest even like lrintf (fcvtns) and saturates,
and since shift >= 0 and INT_MIN + shift <= min, a saturating add followed
by the clamp equals the scalar 64-bit add followed by the clamp. NaN converts
to 0 in both. tests/unit/testdcshift.c checks this over edge values.
*/

/* The reference scalar kernels, also used for the row tails */
static INLINE void opj_dc_shift_row_int_scalar(OPJ_INT32* p, OPJ_UINT32 n,
        OPJ_INT32 shift, OPJ_INT32 min, OPJ_INT32 max)
{
    OPJ_UINT32 i;
    for (i = 0; i < n; ++i) {
        /* 32-bit addition, as upstream */
        p[i] = opj_int_clamp(p[i] + shift, min, max);
    }
}

static INLINE void opj_dc_shift_row_float_scalar(OPJ_INT32* p, OPJ_UINT32 n,
        OPJ_INT32 shift, OPJ_INT32 min, OPJ_INT32 max)
{
    OPJ_UINT32 i;
    for (i = 0; i < n; ++i) {
        OPJ_FLOAT32 l_value;
        memcpy(&l_value, &p[i], sizeof(l_value));
        if (l_value > (OPJ_FLOAT32)INT_MAX) {
            p[i] = max;
        } else if (l_value < INT_MIN) {
            p[i] = min;
        } else {
            /* Do addition on int64 to avoid overflows */
            OPJ_INT64 l_value_int = (OPJ_INT64)opj_lrintf(l_value);
            p[i] = (OPJ_INT32)opj_int64_clamp(l_value_int + shift, min, max);
        }
    }
}

#ifdef __ARM_NEON
#include <arm_neon.h>

static INLINE void opj_dc_shift_row_int(OPJ_INT32* p, OPJ_UINT32 n,
                                        OPJ_INT32 shift, OPJ_INT32 min, OPJ_INT32 max)
{
    const int32x4_t vs = vdupq_n_s32(shift);
    const int32x4_t vmin = vdupq_n_s32(min);
    const int32x4_t vmax = vdupq_n_s32(max);
    OPJ_UINT32 i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t a = vld1q_s32(p + i);
        int32x4_t b = vld1q_s32(p + i + 4);
        /* vaddq_s32 wraps like the scalar 32-bit addition does in practice */
        a = vminq_s32(vmaxq_s32(vaddq_s32(a, vs), vmin), vmax);
        b = vminq_s32(vmaxq_s32(vaddq_s32(b, vs), vmin), vmax);
        vst1q_s32(p + i, a);
        vst1q_s32(p + i + 4, b);
    }
    opj_dc_shift_row_int_scalar(p + i, n - i, shift, min, max);
}

static INLINE void opj_dc_shift_row_float(OPJ_INT32* p, OPJ_UINT32 n,
        OPJ_INT32 shift, OPJ_INT32 min, OPJ_INT32 max)
{
    const int32x4_t vs = vdupq_n_s32(shift);
    const int32x4_t vmin = vdupq_n_s32(min);
    const int32x4_t vmax = vdupq_n_s32(max);
    OPJ_UINT32 i = 0;
    if (shift < 0) {
        /* Not produced by the codec (the shift is 0 or 1 << (prec - 1)); */
        /* the saturating add argument above needs shift >= 0 */
        opj_dc_shift_row_float_scalar(p, n, shift, min, max);
        return;
    }
    for (; i + 8 <= n; i += 8) {
        int32x4_t a = vcvtnq_s32_f32(vld1q_f32((const float*)(p + i)));
        int32x4_t b = vcvtnq_s32_f32(vld1q_f32((const float*)(p + i + 4)));
        a = vminq_s32(vmaxq_s32(vqaddq_s32(a, vs), vmin), vmax);
        b = vminq_s32(vmaxq_s32(vqaddq_s32(b, vs), vmin), vmax);
        vst1q_s32(p + i, a);
        vst1q_s32(p + i + 4, b);
    }
    opj_dc_shift_row_float_scalar(p + i, n - i, shift, min, max);
}

#else

#define opj_dc_shift_row_int opj_dc_shift_row_int_scalar
#define opj_dc_shift_row_float opj_dc_shift_row_float_scalar

#endif

#endif /* OPJ_DC_SHIFT_H */
