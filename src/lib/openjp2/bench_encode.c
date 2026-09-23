/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2026, CtrlMovie AG
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
 * bench_encode: in-memory encode benchmark and zero-copy input self-test.
 *
 * Encodes one frame repeatedly to an in-memory sink so that file I/O and
 * image loading are outside the timed region. Reports the median and minimum
 * wall-clock time of opj_start_compress + opj_encode + opj_end_compress,
 * the codestream size and an FNV-1a 64-bit hash of the codestream bytes.
 *
 * Usage:
 *   bench_encode [-mode cinema|irr|rev] [-threads N] [-iters N]
 *                [-size W H] [-rate R] [-in file.rawl] [-prec P] [-mct 0|1]
 *                [-external] [-mct-domain] [-verify-modes] [-metal]
 *                [-concurrent N] [-check-limits] [-o file.j2k]
 *                [-gray-noise SEED BITS] [-bitrate MBIT]
 *
 * -in expects planar little-endian 16-bit samples, component after component.
 * Without -in a synthetic frame (smooth gradients, blocks and grain) is used.
 * -mode cinema picks OPJ_PROFILE_CINEMA_2K or _4K from the width and the
 * 250 Mbit/s at 24 fps limits; irr is 9/7 with compression ratio -rate;
 * rev is 5/3 lossless.
 * -mct 1 enables the forward multi-component transform (tcp_mct = 1, ICT for
 * 9/7, RCT for 5/3), as libPlayback does. Default 0.
 * -external hands the encoder caller-owned, page-aligned planes through
 * opj_image_set_external_data() on an opj_image_tile_create() image (no
 * OpenJPEG-allocated planes). The planes are allocated once, refilled every
 * iteration and freed by this program.
 * -mct-domain applies the DC level shift and the forward MCT on the CPU,
 * outside the timed region, exactly as opj_tcd_dc_level_shift_encode() and
 * opj_mct_encode_real()/opj_mct_encode() do, and sets INPUT_MCT_DOMAIN=YES.
 * -verify-modes encodes the frame (-iters times each) as normal, -external
 * and -external -mct-domain, checks that the three codestreams are
 * byte-identical, that each decodes (losslessly back to the input for
 * -mode rev), and exercises the ownership rules of
 * opj_image_set_external_data(). Exits non-zero on any mismatch.
 * When built with OPJ_USE_METAL it also encodes the three variants with
 * METAL_T1=YES and requires them to be identical to the CPU codestream
 * (the CPU variants then force METAL_T1=NO); with -metal it additionally
 * fails if the GPU did not encode any code-block.
 * -metal sets METAL_T1=YES (GPU tier-1, OPJ_USE_METAL builds).
 * -check-limits (cinema modes) parses the codestream, sums the tile-part
 * lengths Psot per component (tile-part index modulo the number of
 * components: with the Cinema tile-part split per component, the tile-parts
 * of each progression are in component order) and fails if a component
 * exceeds max_comp_size or the codestream exceeds max_cs_size, as configured
 * (OPJ_CINEMA_24_COMP / OPJ_CINEMA_24_CS unless -bitrate is given).
 * -verify-modes runs it for cinema.
 * -bitrate MBIT (cinema): max_cs_size = MBIT * 1e6 / 8 / 24 and
 * max_comp_size = max_cs_size / 1.25, as the application computes them
 * (j2kDatarate / 8 / editRate at 24 fps).
 * -o writes the codestream of the last iteration to a file.
 * -gray-noise SEED BITS uses a frame of uniform noise of BITS bits around
 * mid-grey, identical in the 3 components: with -mct 1 almost all the bytes
 * go to component 0, so the Cinema per-component limit is the one that
 * binds (instead of the codestream limit).
 * -concurrent N runs N encoder instances in N threads of this process, each
 * doing -iters encodes of the frame (sharing the process-wide Metal
 * context), checks that all codestreams are identical and reports the
 * aggregate frames per second.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

/* The -external planes come from posix_memalign(), as an MTLBuffer's would
 * come from the app, and are released with the system free(). Define the
 * wrapper before opj_includes.h, which poisons free(). */
static void sys_free(void* p)
{
    free(p);
}

#include "opj_includes.h"
#ifdef OPJ_USE_METAL
#include "opj_metal_t1.h"
#endif

#define NUM_COMPS 3

/* ---------------------------------------------------------------------- */
/* In-memory sink with seek support (TLM/PLT markers are patched by seeking
 * back), so that the hash covers the final codestream. */

typedef struct {
    OPJ_BYTE* buf;
    size_t cap;
    size_t len;
    size_t pos;
} mem_sink_t;

static OPJ_BOOL sink_reserve(mem_sink_t* s, size_t need)
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

static OPJ_SIZE_T sink_write(void* p_buffer, OPJ_SIZE_T p_nb_bytes,
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

static OPJ_OFF_T sink_skip(OPJ_OFF_T p_nb_bytes, void* p_user_data)
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

static OPJ_BOOL sink_seek(OPJ_OFF_T p_nb_bytes, void* p_user_data)
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

static OPJ_SIZE_T src_read(void* p_buffer, OPJ_SIZE_T p_nb_bytes,
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

static OPJ_OFF_T src_skip(OPJ_OFF_T p_nb_bytes, void* p_user_data)
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

static OPJ_BOOL src_seek(OPJ_OFF_T p_nb_bytes, void* p_user_data)
{
    mem_src_t* s = (mem_src_t*)p_user_data;
    if (p_nb_bytes < 0 || (size_t)p_nb_bytes > s->len) {
        return OPJ_FALSE;
    }
    s->pos = (size_t)p_nb_bytes;
    return OPJ_TRUE;
}

static OPJ_UINT64 fnv1a64(const OPJ_BYTE* p, size_t n)
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

static void quiet_callback(const char* msg, void* client_data)
{
    (void)msg;
    (void)client_data;
}

static void error_callback(const char* msg, void* client_data)
{
    (void)client_data;
    fprintf(stderr, "[ERROR] %s", msg);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int compare_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static void fill_gray_noise(OPJ_INT32* plane, int w, int h, int prec,
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

static void fill_synthetic(OPJ_INT32* plane, int w, int h, int comp, int prec)
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

static int load_raw(const char* path, OPJ_INT32** planes, int w, int h)
{
    FILE* f = fopen(path, "rb");
    OPJ_UINT16* row;
    int c, x, y;
    if (!f) {
        perror(path);
        return 0;
    }
    row = (OPJ_UINT16*)opj_malloc((size_t)w * sizeof(OPJ_UINT16));
    for (c = 0; c < NUM_COMPS; c++) {
        for (y = 0; y < h; y++) {
            if (fread(row, sizeof(OPJ_UINT16), (size_t)w, f) != (size_t)w) {
                fprintf(stderr, "short read in %s\n", path);
                opj_free(row);
                fclose(f);
                return 0;
            }
            for (x = 0; x < w; x++) {
                planes[c][(size_t)y * w + x] = row[x];
            }
        }
    }
    opj_free(row);
    fclose(f);
    return 1;
}

/* Replicates opj_tcd_dc_level_shift_encode() followed by opj_tcd_mct_encode()
 * (opj_mct_encode_real() for 9/7, opj_mct_encode() for 5/3), with the same
 * float operations in the same order, writing into dst. This is the CPU
 * reference for what the Metal kernel must produce with INPUT_MCT_DOMAIN=YES.
 */
static void to_mct_domain(OPJ_INT32* const* dst, OPJ_INT32* const* src,
                          size_t n, int prec, int reversible, int mct)
{
    const OPJ_INT32 shift = 1 << (prec - 1); /* unsigned components */
    size_t i;
    int c;

    if (reversible) {
        for (c = 0; c < NUM_COMPS; c++) {
            for (i = 0; i < n; i++) {
                dst[c][i] = src[c][i] - shift;
            }
        }
        if (mct) {
            for (i = 0; i < n; i++) {
                OPJ_INT32 r = dst[0][i];
                OPJ_INT32 g = dst[1][i];
                OPJ_INT32 b = dst[2][i];
                dst[0][i] = (r + (g * 2) + b) >> 2;
                dst[1][i] = b - g;
                dst[2][i] = r - g;
            }
        }
    } else {
        OPJ_FLOAT32* f0 = (OPJ_FLOAT32*)dst[0];
        OPJ_FLOAT32* f1 = (OPJ_FLOAT32*)dst[1];
        OPJ_FLOAT32* f2 = (OPJ_FLOAT32*)dst[2];
        for (c = 0; c < NUM_COMPS; c++) {
            OPJ_FLOAT32* f = (OPJ_FLOAT32*)dst[c];
            for (i = 0; i < n; i++) {
                f[i] = (OPJ_FLOAT32)(src[c][i] - shift);
            }
        }
        if (mct) {
            for (i = 0; i < n; i++) {
                OPJ_FLOAT32 r = f0[i];
                OPJ_FLOAT32 g = f1[i];
                OPJ_FLOAT32 b = f2[i];
                OPJ_FLOAT32 y = 0.299f * r + 0.587f * g + 0.114f * b;
                OPJ_FLOAT32 u = -0.16875f * r - 0.331260f * g + 0.5f * b;
                OPJ_FLOAT32 v = 0.5f * r - 0.41869f * g - 0.08131f * b;
                f0[i] = y;
                f1[i] = u;
                f2[i] = v;
            }
        }
    }
}

typedef struct {
    const char* mode;
    int threads;
    int iters;
    int w, h, prec, mct;
    float rate;
    int external;
    int mct_domain;
    int metal; /* -1 option not set, 0 METAL_T1=NO, 1 METAL_T1=YES */
    int max_cs_size;   /* cinema limits */
    int max_comp_size;
} bench_cfg_t;

typedef struct {
    double median_ms;
    double min_ms;
    size_t bytes;
    OPJ_UINT64 hash;
} bench_result_t;

static int setup_params(const bench_cfg_t* cfg, opj_cparameters_t* params)
{
    opj_set_default_encoder_parameters(params);
    params->tcp_numlayers = 1;
    params->cp_disto_alloc = 1;
    params->tcp_rates[0] = 0;
    params->tcp_mct = (char)(cfg->mct ? 1 : 0);
    if (strcmp(cfg->mode, "cinema") == 0) {
        params->rsiz = (cfg->w > 2048) ? OPJ_PROFILE_CINEMA_4K : OPJ_PROFILE_CINEMA_2K;
        params->max_comp_size = cfg->max_comp_size;
        params->max_cs_size = cfg->max_cs_size;
        if (cfg->w > 2048) {
            params->numresolution = 7;
        }
    } else if (strcmp(cfg->mode, "irr") == 0) {
        params->irreversible = 1;
        params->tcp_rates[0] = cfg->rate;
        params->numresolution = 6;
        params->cblockw_init = 32;
        params->cblockh_init = 32;
    } else if (strcmp(cfg->mode, "rev") == 0) {
        params->numresolution = 6;
        params->cblockw_init = 32;
        params->cblockh_init = 32;
    } else {
        fprintf(stderr, "unknown mode %s\n", cfg->mode);
        return 0;
    }
    return 1;
}

static OPJ_INT32* alloc_page_aligned(size_t bytes)
{
    void* p = NULL;
    long page = sysconf(_SC_PAGESIZE);
    if (page < 16) {
        page = 16;
    }
    /* round the size up to whole pages, as an MTLBuffer would be */
    bytes = (bytes + (size_t)page - 1) & ~((size_t)page - 1);
    if (posix_memalign(&p, (size_t)page, bytes) != 0) {
        return NULL;
    }
    return (OPJ_INT32*)p;
}

/* Runs cfg->iters encodes of the master frame. On success returns 1 and, if
 * out_cs is not NULL, hands over the last codestream (caller frees it with
 * opj_free). */
static int run_bench(const bench_cfg_t* cfg, OPJ_INT32* const* master,
                     bench_result_t* res, mem_sink_t* out_cs)
{
    const size_t n = (size_t)cfg->w * (size_t)cfg->h;
    const int reversible = strcmp(cfg->mode, "rev") == 0;
    opj_image_cmptparm_t cmptparm[NUM_COMPS];
    OPJ_INT32* ext[NUM_COMPS] = { NULL, NULL, NULL };
    mem_sink_t sink;
    double* times;
    int i, c, ok_all = 1;

    memset(&sink, 0, sizeof(sink));
    memset(cmptparm, 0, sizeof(cmptparm));
    for (c = 0; c < NUM_COMPS; c++) {
        cmptparm[c].dx = 1;
        cmptparm[c].dy = 1;
        cmptparm[c].w = (OPJ_UINT32)cfg->w;
        cmptparm[c].h = (OPJ_UINT32)cfg->h;
        cmptparm[c].prec = (OPJ_UINT32)cfg->prec;
        cmptparm[c].sgnd = 0;
    }
    if (cfg->external) {
        for (c = 0; c < NUM_COMPS; c++) {
            ext[c] = alloc_page_aligned(sizeof(OPJ_INT32) * n);
            if (!ext[c]) {
                fprintf(stderr, "posix_memalign failed\n");
                return 0;
            }
        }
    }
    /* Pre-size the sink so that growing it is not timed. */
    if (!sink_reserve(&sink, n * NUM_COMPS * 2 + (1u << 20))) {
        fprintf(stderr, "out of memory\n");
        return 0;
    }
    times = (double*)opj_malloc(sizeof(double) * (size_t)cfg->iters);

    for (i = 0; i < cfg->iters; i++) {
        opj_cparameters_t params;
        opj_codec_t* codec;
        opj_stream_t* stream;
        opj_image_t* image;
        OPJ_INT32* planes[NUM_COMPS];
        OPJ_BOOL ok;
        double t0;

        /* A single-tile encode transforms the planes in place and moves them
         * out of the image, so a fresh image is needed for every iteration. */
        if (cfg->external) {
            image = opj_image_tile_create(NUM_COMPS, cmptparm, OPJ_CLRSPC_SRGB);
        } else {
            image = opj_image_create(NUM_COMPS, cmptparm, OPJ_CLRSPC_SRGB);
        }
        if (!image) {
            fprintf(stderr, "image creation failed\n");
            return 0;
        }
        image->x0 = 0;
        image->y0 = 0;
        image->x1 = (OPJ_UINT32)cfg->w;
        image->y1 = (OPJ_UINT32)cfg->h;
        for (c = 0; c < NUM_COMPS; c++) {
            if (cfg->external) {
                if (!opj_image_set_external_data(image, (OPJ_UINT32)c, ext[c])) {
                    fprintf(stderr, "opj_image_set_external_data failed\n");
                    return 0;
                }
            }
            planes[c] = image->comps[c].data;
        }
        if (cfg->mct_domain) {
            to_mct_domain(planes, master, n, cfg->prec, reversible, cfg->mct);
        } else {
            for (c = 0; c < NUM_COMPS; c++) {
                memcpy(planes[c], master[c], sizeof(OPJ_INT32) * n);
            }
        }

        if (!setup_params(cfg, &params)) {
            return 0;
        }
        codec = opj_create_compress(OPJ_CODEC_J2K);
        opj_set_info_handler(codec, quiet_callback, NULL);
        opj_set_warning_handler(codec, quiet_callback, NULL);
        opj_set_error_handler(codec, error_callback, NULL);
        if (!opj_setup_encoder(codec, &params, image)) {
            fprintf(stderr, "opj_setup_encoder failed\n");
            return 0;
        }
        if (cfg->mct_domain || cfg->metal >= 0) {
            const char* opts[3] = { NULL, NULL, NULL };
            int no = 0;
            if (cfg->mct_domain) {
                opts[no++] = "INPUT_MCT_DOMAIN=YES";
            }
            if (cfg->metal >= 0) {
                opts[no++] = cfg->metal ? "METAL_T1=YES" : "METAL_T1=NO";
            }
            if (!opj_encoder_set_extra_options(codec, opts)) {
                fprintf(stderr, "opj_encoder_set_extra_options failed\n");
                return 0;
            }
        }
        if (cfg->threads > 1 && !opj_codec_set_threads(codec, cfg->threads)) {
            fprintf(stderr, "opj_codec_set_threads failed\n");
            return 0;
        }
        stream = opj_stream_create(1 << 20, OPJ_FALSE);
        opj_stream_set_write_function(stream, sink_write);
        opj_stream_set_skip_function(stream, sink_skip);
        opj_stream_set_seek_function(stream, sink_seek);
        opj_stream_set_user_data(stream, &sink, NULL);

        sink.len = 0;
        sink.pos = 0;
        t0 = now_ms();
        ok = opj_start_compress(codec, image, stream) &&
             opj_encode(codec, stream) &&
             opj_end_compress(codec, stream);
        times[i] = now_ms() - t0;

        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        if (!ok) {
            fprintf(stderr, "encode failed at iteration %d\n", i);
            ok_all = 0;
            break;
        }
        res->bytes = sink.len;
        res->hash = fnv1a64(sink.buf, sink.len);
    }

    if (ok_all) {
        qsort(times, (size_t)cfg->iters, sizeof(double), compare_double);
        res->median_ms = times[cfg->iters / 2];
        res->min_ms = times[0];
    }
    opj_free(times);

    /* The planes are ours: OpenJPEG must not have freed them. Touch them
     * (AddressSanitizer flags a use after free here) and free them. */
    for (c = 0; c < NUM_COMPS; c++) {
        if (ext[c]) {
            memset(ext[c], 0, sizeof(OPJ_INT32) * n);
            sys_free(ext[c]);
        }
    }
    if (out_cs && ok_all) {
        *out_cs = sink;
    } else {
        opj_free(sink.buf);
    }
    return ok_all;
}

/* Decodes a codestream and compares it with the master frame: exact for 5/3,
 * PSNR otherwise. Returns 1 when it decodes (and, for 5/3, is lossless). */
static int check_decode(const bench_cfg_t* cfg, const mem_sink_t* cs,
                        OPJ_INT32* const* master)
{
    const size_t n = (size_t)cfg->w * (size_t)cfg->h;
    const int reversible = strcmp(cfg->mode, "rev") == 0;
    opj_dparameters_t dparams;
    opj_codec_t* codec;
    opj_stream_t* stream;
    opj_image_t* image = NULL;
    mem_src_t src;
    int ok, c;

    src.buf = cs->buf;
    src.len = cs->len;
    src.pos = 0;
    opj_set_default_decoder_parameters(&dparams);
    codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_info_handler(codec, quiet_callback, NULL);
    opj_set_warning_handler(codec, quiet_callback, NULL);
    opj_set_error_handler(codec, error_callback, NULL);
    stream = opj_stream_create(1 << 20, OPJ_TRUE);
    opj_stream_set_read_function(stream, src_read);
    opj_stream_set_skip_function(stream, src_skip);
    opj_stream_set_seek_function(stream, src_seek);
    opj_stream_set_user_data(stream, &src, NULL);
    opj_stream_set_user_data_length(stream, cs->len);
    ok = opj_setup_decoder(codec, &dparams) &&
         opj_read_header(stream, codec, &image) &&
         opj_decode(codec, stream, image) &&
         opj_end_decompress(codec, stream);
    if (ok && (image->numcomps != NUM_COMPS ||
               image->comps[0].w != (OPJ_UINT32)cfg->w ||
               image->comps[0].h != (OPJ_UINT32)cfg->h)) {
        fprintf(stderr, "decoded image has unexpected geometry\n");
        ok = 0;
    }
    if (ok) {
        double se = 0.0, maxv = (double)((1 << cfg->prec) - 1);
        size_t mismatches = 0, i;
        for (c = 0; c < NUM_COMPS; c++) {
            for (i = 0; i < n; i++) {
                double d = (double)image->comps[c].data[i] - (double)master[c][i];
                se += d * d;
                if (d != 0.0) {
                    mismatches++;
                }
            }
        }
        if (reversible) {
            printf("  decode: %zu mismatching samples (lossless expected)\n", mismatches);
            if (mismatches) {
                ok = 0;
            }
        } else {
            double mse = se / (double)(n * NUM_COMPS);
            double psnr = mse > 0 ? 10.0 * log10(maxv * maxv / mse) : 999.0;
            printf("  decode: PSNR %.2f dB\n", psnr);
            if (psnr < 30.0) {
                ok = 0;
            }
        }
    } else {
        fprintf(stderr, "decoding the codestream failed\n");
    }
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    return ok;
}

/* Ownership rules of opj_image_set_external_data(), checked under ASan. */
static int check_ownership(void)
{
    opj_image_cmptparm_t parm;
    opj_image_t* image;
    OPJ_INT32* plane = alloc_page_aligned(64 * 64 * sizeof(OPJ_INT32));
    OPJ_INT32* plane2 = alloc_page_aligned(64 * 64 * sizeof(OPJ_INT32));
    int ok = 1;

    memset(&parm, 0, sizeof(parm));
    parm.dx = parm.dy = 1;
    parm.w = parm.h = 64;
    parm.prec = 12;
    /* opj_image_create allocates a plane; registering must free it. */
    image = opj_image_create(1, &parm, OPJ_CLRSPC_GRAY);
    if (!image || !plane || !plane2) {
        return 0;
    }
    ok &= opj_image_set_external_data(image, 0, plane) == OPJ_TRUE;
    ok &= image->comps[0].data == plane && image->comps[0].data_is_external;
    /* replacing an external plane must not free the previous one */
    ok &= opj_image_set_external_data(image, 0, plane2) == OPJ_TRUE;
    ok &= opj_image_set_external_data(image, 0, plane) == OPJ_TRUE;
    /* invalid arguments */
    ok &= opj_image_set_external_data(image, 1, plane) == OPJ_FALSE;
    ok &= opj_image_set_external_data(image, 0, plane + 1) == OPJ_FALSE;
    ok &= opj_image_set_external_data(NULL, 0, plane) == OPJ_FALSE;
    ok &= image->comps[0].data == plane;
    opj_image_destroy(image); /* must not free plane */
    /* detaching with NULL */
    image = opj_image_tile_create(1, &parm, OPJ_CLRSPC_GRAY);
    ok &= opj_image_set_external_data(image, 0, plane2) == OPJ_TRUE;
    ok &= opj_image_set_external_data(image, 0, NULL) == OPJ_TRUE;
    ok &= image->comps[0].data == NULL && !image->comps[0].data_is_external;
    opj_image_destroy(image);
    /* both planes still ours */
    memset(plane, 0, 64 * 64 * sizeof(OPJ_INT32));
    memset(plane2, 0, 64 * 64 * sizeof(OPJ_INT32));
    sys_free(plane);
    sys_free(plane2);
    printf("  ownership checks: %s\n", ok ? "ok" : "FAILED");
    return ok;
}

/* Multi-tile encode with external planes: the tile data is copied, so the
 * planes must come back untouched and must not be freed by OpenJPEG. */
static int check_multitile_external(void)
{
    const int w = 96, h = 80;
    const size_t n = (size_t)w * h;
    opj_image_cmptparm_t parm[NUM_COMPS];
    OPJ_INT32* ext[NUM_COMPS];
    opj_cparameters_t params;
    opj_codec_t* codec;
    opj_stream_t* stream;
    opj_image_t* image;
    mem_sink_t sink;
    size_t i;
    int c, ok = 1;

    memset(parm, 0, sizeof(parm));
    memset(&sink, 0, sizeof(sink));
    image = NULL;
    for (c = 0; c < NUM_COMPS; c++) {
        parm[c].dx = parm[c].dy = 1;
        parm[c].w = (OPJ_UINT32)w;
        parm[c].h = (OPJ_UINT32)h;
        parm[c].prec = 12;
        ext[c] = alloc_page_aligned(n * sizeof(OPJ_INT32));
        if (!ext[c]) {
            return 0;
        }
        for (i = 0; i < n; i++) {
            ext[c][i] = (OPJ_INT32)((i * 7 + (size_t)c * 1000) & 4095);
        }
    }
    image = opj_image_tile_create(NUM_COMPS, parm, OPJ_CLRSPC_SRGB);
    if (!image) {
        return 0;
    }
    image->x1 = (OPJ_UINT32)w;
    image->y1 = (OPJ_UINT32)h;
    for (c = 0; c < NUM_COMPS; c++) {
        ok &= opj_image_set_external_data(image, (OPJ_UINT32)c, ext[c]) == OPJ_TRUE;
    }
    opj_set_default_encoder_parameters(&params);
    params.tcp_numlayers = 1;
    params.cp_disto_alloc = 1;
    params.numresolution = 3;
    params.tcp_mct = 1;
    params.irreversible = 1;
    params.tile_size_on = OPJ_TRUE;
    params.cp_tdx = params.cp_tdy = 32;
    codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_info_handler(codec, quiet_callback, NULL);
    opj_set_warning_handler(codec, quiet_callback, NULL);
    opj_set_error_handler(codec, error_callback, NULL);
    stream = opj_stream_create(1 << 16, OPJ_FALSE);
    opj_stream_set_write_function(stream, sink_write);
    opj_stream_set_skip_function(stream, sink_skip);
    opj_stream_set_seek_function(stream, sink_seek);
    opj_stream_set_user_data(stream, &sink, NULL);
    ok &= opj_setup_encoder(codec, &params, image) &&
          opj_start_compress(codec, image, stream) &&
          opj_encode(codec, stream) &&
          opj_end_compress(codec, stream);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    ok &= sink.len > 0;
    for (c = 0; c < NUM_COMPS; c++) {
        for (i = 0; i < n; i++) {
            if (ext[c][i] != (OPJ_INT32)((i * 7 + (size_t)c * 1000) & 4095)) {
                ok = 0;
                break;
            }
        }
        sys_free(ext[c]);
    }
    opj_free(sink.buf);
    printf("  multi-tile external checks: %s\n", ok ? "ok" : "FAILED");
    return ok;
}

/* INPUT_MCT_DOMAIN=YES must be rejected for setups it cannot support. */
static int check_option_rejection(void)
{
    static const char* yes[] = { "INPUT_MCT_DOMAIN=YES", NULL };
    static const char* bad[] = { "INPUT_MCT_DOMAIN=MAYBE", NULL };
    opj_image_cmptparm_t parm[NUM_COMPS];
    opj_image_t* image;
    int k, c, ok = 1;

    memset(parm, 0, sizeof(parm));
    for (c = 0; c < NUM_COMPS; c++) {
        parm[c].dx = parm[c].dy = 1;
        parm[c].w = parm[c].h = 64;
        parm[c].prec = 12;
    }
    image = opj_image_create(NUM_COMPS, parm, OPJ_CLRSPC_SRGB);
    if (!image) {
        return 0;
    }
    image->x1 = image->y1 = 64;
    for (k = 0; k < 4; k++) {
        opj_cparameters_t params;
        opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
        OPJ_FLOAT32 matrix[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        OPJ_INT32 dc_shift[3] = { 0, 0, 0 };
        OPJ_BOOL expect, got;

        opj_set_info_handler(codec, quiet_callback, NULL);
        opj_set_warning_handler(codec, quiet_callback, NULL);
        opj_set_error_handler(codec, quiet_callback, NULL);
        opj_set_default_encoder_parameters(&params);
        params.tcp_numlayers = 1;
        params.cp_disto_alloc = 1;
        params.numresolution = 3;
        params.tcp_mct = 1;
        if (k == 1) { /* multi-tile */
            params.tile_size_on = OPJ_TRUE;
            params.cp_tdx = params.cp_tdy = 32;
        } else if (k == 2) { /* custom MCT */
            opj_set_MCT(&params, matrix, dc_shift, NUM_COMPS);
        }
        if (!opj_setup_encoder(codec, &params, image)) {
            ok = 0;
        } else {
            got = opj_encoder_set_extra_options(codec, k == 3 ? bad : yes);
            expect = (k == 0) ? OPJ_TRUE : OPJ_FALSE;
            if (got != expect) {
                fprintf(stderr, "INPUT_MCT_DOMAIN case %d: got %d, expected %d\n",
                        k, got, expect);
                ok = 0;
            }
        }
        if (params.mct_data) {
            opj_free(params.mct_data);
        }
        opj_destroy_codec(codec);
    }
    opj_image_destroy(image);
    printf("  option rejection checks: %s\n", ok ? "ok" : "FAILED");
    return ok;
}

static OPJ_UINT32 be16(const OPJ_BYTE* p)
{
    return ((OPJ_UINT32)p[0] << 8) | p[1];
}

static OPJ_UINT32 be32(const OPJ_BYTE* p)
{
    return ((OPJ_UINT32)p[0] << 24) | ((OPJ_UINT32)p[1] << 16) |
           ((OPJ_UINT32)p[2] << 8) | p[3];
}

/* -check-limits: SMPTE 429-2 sizes of a Cinema codestream (single tile) */
static int check_limits(const mem_sink_t* cs, int ncomps, int max_comp,
                        int max_cs)
{
    const OPJ_BYTE* b = cs->buf;
    const size_t n = cs->len;
    size_t pos = 2;
    OPJ_UINT64 sums[NUM_COMPS] = { 0, 0, 0 };
    int ntp = 0, ok = 1, c;

    if (n < 4 || be16(b) != 0xFF4F) {
        fprintf(stderr, "check-limits: no SOC\n");
        return 0;
    }
    while (pos + 4 <= n) {
        const OPJ_UINT32 m = be16(b + pos);
        if (m == 0xFF90) { /* SOT: Lsot Isot Psot TPsot TNsot */
            OPJ_UINT32 psot, tpsot;
            if (pos + 12 > n) {
                ok = 0;
                break;
            }
            psot = be32(b + pos + 6);
            tpsot = b[pos + 10];
            if (psot == 0) { /* last tile-part, up to EOC */
                psot = (OPJ_UINT32)(n - 2 - pos);
            }
            sums[tpsot % (OPJ_UINT32)ncomps] += psot;
            ntp++;
            pos += psot;
        } else if (m == 0xFFD9) { /* EOC */
            break;
        } else if ((m & 0xFF00) == 0xFF00) {
            pos += 2 + be16(b + pos + 2);
        } else {
            fprintf(stderr, "check-limits: bad marker 0x%04x at %zu\n", m, pos);
            return 0;
        }
    }
    printf("  limits: %d tile-parts, per component", ntp);
    for (c = 0; c < ncomps; c++) {
        printf(" %llu", (unsigned long long)sums[c]);
        if (sums[c] > (OPJ_UINT64)max_comp) {
            ok = 0;
        }
    }
    if (n > (size_t)max_cs) {
        ok = 0;
    }
    printf(" (max %d), codestream %zu (max %d, slack %ld): %s\n", max_comp, n,
           max_cs, (long)max_cs - (long)n, ok ? "ok" : "EXCEEDED");
    return ok;
}

static void print_result(const bench_cfg_t* cfg, const char* label,
                         const bench_result_t* r)
{
    printf("%s %dx%d prec=%d mct=%d threads=%d iters=%d%s%s: median %.1f ms, "
           "min %.1f ms, %zu bytes, hash %016llx\n",
           cfg->mode, cfg->w, cfg->h, cfg->prec, cfg->mct, cfg->threads,
           cfg->iters, label, cfg->metal == 1 ? " metal" : "", r->median_ms,
           r->min_ms, r->bytes, (unsigned long long)r->hash);
}

typedef struct {
    const bench_cfg_t* cfg;
    OPJ_INT32* const* master;
    bench_result_t res;
    int ok;
} bench_thread_t;

static void* bench_thread(void* arg)
{
    bench_thread_t* t = (bench_thread_t*)arg;
    memset(&t->res, 0, sizeof(t->res));
    t->ok = run_bench(t->cfg, t->master, &t->res, NULL);
    return NULL;
}

/* -concurrent N: N encoder instances in N threads */
static int run_concurrent(const bench_cfg_t* cfg, OPJ_INT32* const* master,
                          int n)
{
    pthread_t* th = (pthread_t*)opj_malloc(sizeof(pthread_t) * (size_t)n);
    bench_thread_t* t = (bench_thread_t*)opj_malloc(sizeof(bench_thread_t) *
                        (size_t)n);
    double t0, dt;
    int i, ok = 1;
    bench_result_t ref;
    if (!th || !t) {
        return 0;
    }
    memset(&ref, 0, sizeof(ref));
    if (cfg->metal == 1) {
        /* CPU reference codestream (one frame, untimed) */
        bench_cfg_t c = *cfg;
        c.metal = 0;
        c.iters = 1;
        if (!run_bench(&c, master, &ref, NULL)) {
            return 0;
        }
    }
    t0 = now_ms();
    for (i = 0; i < n; i++) {
        t[i].cfg = cfg;
        t[i].master = master;
        if (pthread_create(&th[i], NULL, bench_thread, &t[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 0;
        }
    }
    for (i = 0; i < n; i++) {
        pthread_join(th[i], NULL);
    }
    dt = now_ms() - t0;
    for (i = 0; i < n; i++) {
        if (!t[i].ok || t[i].res.hash != t[0].res.hash ||
                t[i].res.bytes != t[0].res.bytes) {
            fprintf(stderr, "MISMATCH: instance %d (ok %d, hash %016llx)\n", i,
                    t[i].ok, (unsigned long long)t[i].res.hash);
            ok = 0;
        }
    }
    if (cfg->metal == 1 && (ref.hash != t[0].res.hash ||
                            ref.bytes != t[0].res.bytes)) {
        fprintf(stderr, "MISMATCH: metal %016llx vs CPU reference %016llx\n",
                (unsigned long long)t[0].res.hash, (unsigned long long)ref.hash);
        ok = 0;
    }
    print_result(cfg, " instance 0", &t[0].res);
#ifdef OPJ_USE_METAL
    if (cfg->metal == 1) {
        OPJ_UINT32 nb = 0;
        OPJ_SIZE_T bytes = opj_metal_t1_pool_bytes(&nb);
        printf("metal batch pool: %u batches, %.1f MB (%.1f MB per batch)\n",
               nb, (double)bytes / 1e6, nb ? (double)bytes / 1e6 / nb : 0.0);
    }
#endif
    if (cfg->metal == 1) {
        printf("CPU reference hash %016llx: %s\n", (unsigned long long)ref.hash,
               ref.hash == t[0].res.hash ? "match" : "MISMATCH");
    }
    printf("concurrent %d instances x %d frames: %.1f s wall, %.2f fps "
           "aggregate, codestreams %s\n", n, cfg->iters, dt / 1e3,
           (double)n * cfg->iters / (dt / 1e3),
           ok ? "identical" : "DIFFER");
    opj_free(th);
    opj_free(t);
    return ok;
}

int main(int argc, char** argv)
{
    const char* infile = NULL;
    bench_cfg_t cfg;
    OPJ_INT32* master[NUM_COMPS];
    int verify = 0;
    int concurrent = 0;
    int check = 0;
    const char* outfile = NULL;
    int gray_seed = -1, gray_bits = 10;
    int i, c;

    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = "cinema";
    cfg.threads = 1;
    cfg.iters = 5;
    cfg.w = 4096;
    cfg.h = 2160;
    cfg.prec = 12;
    cfg.rate = 20.0f;
    cfg.metal = -1;
    cfg.max_cs_size = OPJ_CINEMA_24_CS;
    cfg.max_comp_size = OPJ_CINEMA_24_COMP;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-iters") == 0 && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-size") == 0 && i + 2 < argc) {
            cfg.w = atoi(argv[++i]);
            cfg.h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-rate") == 0 && i + 1 < argc) {
            cfg.rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-in") == 0 && i + 1 < argc) {
            infile = argv[++i];
        } else if (strcmp(argv[i], "-prec") == 0 && i + 1 < argc) {
            cfg.prec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-mct") == 0 && i + 1 < argc) {
            cfg.mct = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "-external") == 0) {
            cfg.external = 1;
        } else if (strcmp(argv[i], "-mct-domain") == 0) {
            cfg.mct_domain = 1;
        } else if (strcmp(argv[i], "-verify-modes") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-metal") == 0) {
            cfg.metal = 1;
        } else if (strcmp(argv[i], "-check-limits") == 0) {
            check = 1;
        } else if (strcmp(argv[i], "-bitrate") == 0 && i + 1 < argc) {
            /* as the application: j2kDatarate / 8 / editRate, then / 1.25 */
            const long long datarate = (long long)(atof(argv[++i]) * 1e6);
            cfg.max_cs_size = (int)(datarate / 8 / 24);
            cfg.max_comp_size = (int)(cfg.max_cs_size / 1.25);
        } else if (strcmp(argv[i], "-gray-noise") == 0 && i + 2 < argc) {
            gray_seed = atoi(argv[++i]);
            gray_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outfile = argv[++i];
        } else if (strcmp(argv[i], "-concurrent") == 0 && i + 1 < argc) {
            concurrent = atoi(argv[++i]);
        } else {
            fprintf(stderr,
                    "usage: bench_encode [-mode cinema|irr|rev] [-threads N] [-iters N] "
                    "[-size W H] [-rate R] [-in file.rawl] [-prec P] [-mct 0|1] "
                    "[-external] [-mct-domain] [-verify-modes] [-metal] "
                    "[-concurrent N] [-check-limits] [-o file.j2k] "
                    "[-gray-noise SEED BITS] [-bitrate MBIT]\n");
            return 1;
        }
    }
    if (cfg.iters < 1) {
        cfg.iters = 1;
    }
    if (cfg.w < 1 || cfg.h < 1 || cfg.prec < 1 || cfg.prec > 16) {
        fprintf(stderr, "invalid size or precision\n");
        return 1;
    }

    for (c = 0; c < NUM_COMPS; c++) {
        master[c] = (OPJ_INT32*)opj_malloc(sizeof(OPJ_INT32) * (size_t)cfg.w * cfg.h);
        if (!master[c]) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
    }
    if (infile) {
        if (!load_raw(infile, master, cfg.w, cfg.h)) {
            return 1;
        }
    } else if (gray_seed >= 0) {
        if (gray_bits < 1 || gray_bits > cfg.prec) {
            fprintf(stderr, "invalid -gray-noise bits\n");
            return 1;
        }
        for (c = 0; c < NUM_COMPS; c++) {
            fill_gray_noise(master[c], cfg.w, cfg.h, cfg.prec, (unsigned)gray_seed,
                            gray_bits);
        }
    } else {
        for (c = 0; c < NUM_COMPS; c++) {
            fill_synthetic(master[c], cfg.w, cfg.h, c, cfg.prec);
        }
    }
    setvbuf(stdout, NULL, _IONBF, 0);

    if (verify) {
        static const char* labels[6] = { " normal", " external",
                                         " external+mct-domain",
                                         " normal", " external",
                                         " external+mct-domain"
                                       };
        bench_result_t r[6];
        mem_sink_t cs[6];
        int k, ok = 1;
#ifdef OPJ_USE_METAL
        const int nvariants = 6;
        const OPJ_UINT64 gpu_before = opj_metal_t1_gpu_cblk_count();
#else
        const int nvariants = 3;
#endif
        for (k = 0; k < nvariants; k++) {
            bench_cfg_t v = cfg;
            v.external = (k % 3) >= 1;
            v.mct_domain = (k % 3) == 2;
#ifdef OPJ_USE_METAL
            v.metal = k >= 3 ? 1 : 0;
#endif
            memset(&r[k], 0, sizeof(r[k]));
            memset(&cs[k], 0, sizeof(cs[k]));
            if (!run_bench(&v, master, &r[k], &cs[k])) {
                return 1;
            }
            print_result(&v, labels[k], &r[k]);
            if (!check_decode(&v, &cs[k], master)) {
                ok = 0;
            }
            if (strcmp(cfg.mode, "cinema") == 0 &&
                    !check_limits(&cs[k], NUM_COMPS, cfg.max_comp_size, cfg.max_cs_size)) {
                ok = 0;
            }
        }
        for (k = 1; k < nvariants; k++) {
            if (r[k].bytes != r[0].bytes || r[k].hash != r[0].hash ||
                    memcmp(cs[k].buf, cs[0].buf, cs[0].len) != 0) {
                size_t j, lim = cs[k].len < cs[0].len ? cs[k].len : cs[0].len;
                for (j = 0; j < lim && cs[k].buf[j] == cs[0].buf[j]; j++) {
                }
                fprintf(stderr, "MISMATCH:%s%s differs from normal (first differing byte %zu)\n",
                        labels[k], k >= 3 ? " metal" : "", j);
                ok = 0;
            }
        }
        for (k = 0; k < nvariants; k++) {
            opj_free(cs[k].buf);
        }
#ifdef OPJ_USE_METAL
        {
            const OPJ_UINT64 gpu = opj_metal_t1_gpu_cblk_count() - gpu_before;
            printf("  metal: %llu code-blocks encoded on the GPU\n",
                   (unsigned long long)gpu);
            if (cfg.metal == 1 && gpu == 0) {
                fprintf(stderr, "-metal: the GPU was not used\n");
                ok = 0;
            }
        }
#endif
        ok &= check_ownership();
        ok &= check_option_rejection();
        ok &= check_multitile_external();
        printf("verify-modes: %s\n", ok ? "PASS (codestreams identical)" : "FAIL");
        for (c = 0; c < NUM_COMPS; c++) {
            opj_free(master[c]);
        }
        return ok ? 0 : 1;
    } else if (concurrent > 0) {
        int ok = run_concurrent(&cfg, master, concurrent);
        for (c = 0; c < NUM_COMPS; c++) {
            opj_free(master[c]);
        }
        return ok ? 0 : 1;
    } else {
        bench_result_t r;
        mem_sink_t cs;
        memset(&r, 0, sizeof(r));
        memset(&cs, 0, sizeof(cs));
        if (!run_bench(&cfg, master, &r, (check || outfile) ? &cs : NULL)) {
            return 1;
        }
        if (outfile) {
            FILE* f = fopen(outfile, "wb");
            if (!f || fwrite(cs.buf, 1, cs.len, f) != cs.len) {
                fprintf(stderr, "cannot write %s\n", outfile);
                return 1;
            }
            fclose(f);
        }
        print_result(&cfg, cfg.mct_domain ? (cfg.external ? " external+mct-domain" :
                     " mct-domain") : (cfg.external ? " external" : ""), &r);
        if (check || outfile) {
            int ok = check ? check_limits(&cs, NUM_COMPS, cfg.max_comp_size,
                                          cfg.max_cs_size) : 1;
            opj_free(cs.buf);
            if (!ok) {
                for (c = 0; c < NUM_COMPS; c++) {
                    opj_free(master[c]);
                }
                return 1;
            }
        }
    }

    for (c = 0; c < NUM_COMPS; c++) {
        opj_free(master[c]);
    }
    return 0;
}
