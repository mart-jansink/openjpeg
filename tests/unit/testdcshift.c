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

/* Checks that the NEON DC level shift row kernels (dc_shift.h) give exactly
 * the results of the scalar ones, over edge values and random data. */

#include "opj_includes.h"
#include "dc_shift.h"

#include <math.h>

static OPJ_UINT32 g_rng = 12345;
static OPJ_UINT32 rnd(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

#define N 1027 /* not a multiple of 8, so the tails run too */

static int check(const OPJ_INT32* in, int is_float, OPJ_INT32 shift,
                 OPJ_INT32 min, OPJ_INT32 max, const char* what)
{
    OPJ_INT32 a[N], b[N];
    int i;
    memcpy(a, in, sizeof(a));
    memcpy(b, in, sizeof(b));
    if (is_float) {
        opj_dc_shift_row_float_scalar(a, N, shift, min, max);
        opj_dc_shift_row_float(b, N, shift, min, max);
    } else {
        opj_dc_shift_row_int_scalar(a, N, shift, min, max);
        opj_dc_shift_row_int(b, N, shift, min, max);
    }
    for (i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            float f;
            memcpy(&f, &in[i], sizeof(f));
            fprintf(stderr, "%s: %s shift %d [%d, %d] sample %d (0x%08x, %g): "
                    "scalar %d, vector %d\n", what, is_float ? "float" : "int",
                    shift, min, max, i, (unsigned)in[i], f, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    static const float edge_f[] = {
        0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 0.49999997f,
        -0.49999997f, 1.0f, -1.0f, 127.5f, 128.5f, 2047.5f, -2048.5f,
        32767.5f, 32768.5f, -32768.5f, 65535.5f, 8388607.5f, 16777216.0f,
        2147483520.0f, 2147483648.0f, 2147483904.0f, -2147483648.0f,
        -2147483904.0f, 4294967296.0f, -4294967296.0f, 1e30f, -1e30f,
        1e-30f, -1e-30f, 1.4e-45f
    };
    OPJ_INT32 in[N];
    int prec, sgnd, rep, i, ok = 1, ncheck = 0;
    for (prec = 1; prec <= 31 && ok; prec++) {
        for (sgnd = 0; sgnd <= 1 && ok; sgnd++) {
            OPJ_INT32 min, max, shift;
            OPJ_INT64 range;
            if (sgnd) {
                min = -(1 << (prec - 1));
                max = (1 << (prec - 1)) - 1;
                shift = 0;
            } else {
                min = 0;
                max = (OPJ_INT32)((1U << prec) - 1);
                shift = 1 << (prec - 1);
            }
            range = (OPJ_INT64)max - min;
            for (rep = 0; rep < 8 && ok; rep++) {
                /* floats: edge values, specials and random values around */
                /* the range, mixed in random order */
                for (i = 0; i < N; i++) {
                    float f;
                    const OPJ_UINT32 r = rnd();
                    switch (r % 6) {
                    case 0:
                        f = edge_f[rnd() % (sizeof(edge_f) / sizeof(edge_f[0]))];
                        if (rnd() & 1) {
                            f = -f;
                        }
                        break;
                    case 1:
                        f = (rnd() & 1) ? INFINITY : -INFINITY;
                        break;
                    case 2:
                        f = NAN;
                        if (rnd() & 1) {
                            f = -f;
                        }
                        break;
                    case 3: {
                        /* any bit pattern */
                        OPJ_UINT32 u = rnd();
                        memcpy(&f, &u, sizeof(f));
                        break;
                    }
                    case 4:
                        /* exact halves near the range */
                        f = (float)((OPJ_INT64)((OPJ_UINT64)rnd() % (OPJ_UINT64)(2 * range + 9)) -
                                    range - 4) * 0.5f + (float)min;
                        break;
                    default:
                        f = ((float)(rnd() % 2000001) / 1000000.0f - 1.0f) *
                            (float)range * 1.25f;
                        break;
                    }
                    memcpy(&in[i], &f, sizeof(f));
                }
                ok = check(in, 1, shift, min, max, "random") && ok;
                ncheck++;
                /* ints: random values around the range and extremes */
                for (i = 0; i < N; i++) {
                    const OPJ_UINT32 r = rnd();
                    switch (r % 4) {
                    case 0:
                        in[i] = (r & 4) ? INT_MAX : INT_MIN;
                        break;
                    case 1:
                        in[i] = (OPJ_INT32)rnd();
                        break;
                    default:
                        in[i] = (OPJ_INT32)((OPJ_INT64)min - shift - range / 4 +
                                            (OPJ_INT64)((OPJ_UINT64)rnd() % (OPJ_UINT64)(range * 3 / 2 + 3)));
                        break;
                    }
                }
                ok = check(in, 0, shift, min, max, "random") && ok;
                ncheck++;
            }
        }
    }
    printf("%d row checks: %s\n", ncheck, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
