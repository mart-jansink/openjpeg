/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2026, CtrlMovie AG
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


/*
 * Helpers shared by bench_encode.c and bench_decode.c: in-memory streams,
 * hashing, timing and synthetic frames. Include after opj_includes.h.
 */

#ifndef OPJ_BENCH_UTIL_H
#define OPJ_BENCH_UTIL_H

#include <math.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------------- */
/* In-memory sink with seek support (TLM/PLT markers are patched by seeking
 * back), so that the hash covers the final codestream. */

typedef struct {
    OPJ_BYTE* buf;
    size_t cap;
    size_t len;
    size_t pos;
} mem_sink_t;

static INLINE OPJ_BOOL sink_reserve(mem_sink_t* s, size_t need)
{
    if (need > s->cap) {
        size_t ncap = s->cap ? s->cap : (1u << 20);
        OPJ_BYTE* nbuf;
        while (ncap < need) {
            ncap *= 2;
        }
        nbuf = (OPJ_BYTE*)opj_realloc(s->buf, ncap);
        if (!nbuf) {
            return OPJ_FALSE;
        }
        s->buf = nbuf;
        s->cap = ncap;
    }
    if (need > s->len) {
        memset(s->buf + s->len, 0, need - s->len);
        s->len = need;
    }
    return OPJ_TRUE;
}

static INLINE OPJ_SIZE_T sink_write(void* p_buffer, OPJ_SIZE_T p_nb_bytes,
                             void* p_user_data)
{
    mem_sink_t* s = (mem_sink_t*)p_user_data;
    if (!sink_reserve(s, s->pos + p_nb_bytes)) {
        return (OPJ_SIZE_T) - 1;
    }
    memcpy(s->buf + s->pos, p_buffer, p_nb_bytes);
    s->pos += p_nb_bytes;
    return p_nb_bytes;
}

static INLINE OPJ_OFF_T sink_skip(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    mem_sink_t* s = (mem_sink_t*)p_user_data;
    if (p_nb_bytes < 0 && (size_t)(-p_nb_bytes) > s->pos) {
        return -1;
    }
    if (p_nb_bytes > 0 && !sink_reserve(s, s->pos + (size_t)p_nb_bytes)) {
        return -1;
    }
    s->pos = (size_t)((OPJ_OFF_T)s->pos + p_nb_bytes);
    return p_nb_bytes;
}

static INLINE OPJ_BOOL sink_seek(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    mem_sink_t* s = (mem_sink_t*)p_user_data;
    if (p_nb_bytes < 0 || !sink_reserve(s, (size_t)p_nb_bytes)) {
        return OPJ_FALSE;
    }
    s->pos = (size_t)p_nb_bytes;
    return OPJ_TRUE;
}

/* Read side, used by -verify-modes to decode the produced codestream. */
typedef struct {
    const OPJ_BYTE* buf;
    size_t len;
    size_t pos;
} mem_src_t;

static INLINE OPJ_SIZE_T src_read(void* p_buffer, OPJ_SIZE_T p_nb_bytes,
                           void* p_user_data)
{
    mem_src_t* s = (mem_src_t*)p_user_data;
    size_t n = s->len - s->pos;
    if (n == 0) {
        return (OPJ_SIZE_T) - 1;
    }
    if (n > p_nb_bytes) {
        n = p_nb_bytes;
    }
    memcpy(p_buffer, s->buf + s->pos, n);
    s->pos += n;
    return n;
}

static INLINE OPJ_OFF_T src_skip(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    mem_src_t* s = (mem_src_t*)p_user_data;
    OPJ_OFF_T np = (OPJ_OFF_T)s->pos + p_nb_bytes;
    if (np < 0) {
        np = 0;
    }
    if ((size_t)np > s->len) {
        np = (OPJ_OFF_T)s->len;
    }
    p_nb_bytes = np - (OPJ_OFF_T)s->pos;
    s->pos = (size_t)np;
    return p_nb_bytes;
}

static INLINE OPJ_BOOL src_seek(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    mem_src_t* s = (mem_src_t*)p_user_data;
    if (p_nb_bytes < 0 || (size_t)p_nb_bytes > s->len) {
        return OPJ_FALSE;
    }
    s->pos = (size_t)p_nb_bytes;
    return OPJ_TRUE;
}

static INLINE OPJ_UINT64 fnv1a64(const OPJ_BYTE* p, size_t n)
{
    OPJ_UINT64 h = 14695981039346656037ULL;
    size_t i;
    for (i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ---------------------------------------------------------------------- */

static INLINE void quiet_callback(const char* msg, void* client_data)
{
    (void)msg;
    (void)client_data;
}

static INLINE void error_callback(const char* msg, void* client_data)
{
    (void)client_data;
    fprintf(stderr, "[ERROR] %s", msg);
}

static INLINE double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static INLINE int compare_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static INLINE void fill_gray_noise(OPJ_INT32* plane, int w, int h, int prec,
                            unsigned seed, int bits)
{
    OPJ_UINT32 s = 2463534242u ^ (seed * 2654435761u);
    const OPJ_INT32 base = (1 << (prec - 1)) - (1 << (bits - 1));
    size_t i, n = (size_t)w * (size_t)h;
    for (i = 0; i < n; i++) {
        /* xorshift32 */
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        plane[i] = base + (OPJ_INT32)(s >> (32 - bits));
    }
}

static INLINE void fill_synthetic(OPJ_INT32* plane, int w, int h, int comp, int prec)
{
    OPJ_UINT32 s = 12345u + (OPJ_UINT32)comp;
    double maxv = (double)((1 << prec) - 1);
    double mid = maxv / 2;
    int x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            double v = mid + (mid * 0.44) * sin(x * 0.0021 + comp) * cos(y * 0.0017 + 0.5 * comp)
                       + (mid * 0.24) * sin((x + y) * 0.011);
            double n;
            if ((((x / 300) + (y / 220)) % 5) == 0) {
                v += mid * 0.15;
            }
            s = s * 1664525u + 1013904223u;
            n = ((s >> 8) & 0xFF) / 255.0 - 0.5;
            v += n * (mid * 0.02);
            if (v < 0) {
                v = 0;
            }
            if (v > maxv) {
                v = maxv;
            }
            plane[(size_t)y * w + x] = (OPJ_INT32)v;
        }
    }
}

#endif /* OPJ_BENCH_UTIL_H */
