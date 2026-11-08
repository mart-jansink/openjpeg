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


/*
 * bench_decode: in-memory decode benchmark and GPU tier-1 decoder verifier.
 *
 * The codestream is either a synthetic frame encoded in memory (on the CPU)
 * or read from files given with -i. Decoding happens from memory, so file
 * I/O is outside the timed region.
 *
 * Usage:
 *   bench_decode [-mode cinema|irr|rev] [-mct 0|1] [-size W H] [-prec P]
 *                [-rate R] [-bitrate MBIT] [-cblk N] [-layers N]
 *                [-cblksty N] [-gray-noise SEED BITS] [-o file.j2k]
 *                [-i file.j2c ...]
 *                [-threads N] [-iters N] [-reduce R] [-area X0 Y0 X1 Y1]
 *                [-metal | -cpu] [-verify] [-require-gpu]
 *                [-concurrent N] [-output-mct-domain]
 *
 * Without -verify it times -iters decodes of each codestream (median and
 * minimum wall time of opj_read_header + opj_decode + opj_end_decompress)
 * and prints an FNV-1a hash of the decoded planes. -metal sets the decoder
 * option METAL_T1=YES, -cpu METAL_T1=NO (default: option not set, so the
 * OPJ_METAL_T1_DECODE environment variable decides).
 *
 * -verify decodes each codestream with METAL_T1=NO and METAL_T1=YES for
 * reduce 0, 1 and 2, single-threaded and with -threads N (default 4), as
 * the whole image and as a window (opj_set_decode_area(), the partial
 * decoding path of the CPU decoder), and requires every decoded plane to be
 * bit-identical. The METAL_T1=NO reference uses the original scalar DC level
 * shift (OPJ_DEC_SCALAR_DC_SHIFT=1), so the NEON one is checked as well,
 * and the Metal decodes use the fixed CPU/GPU split formula
 * (OPJ_METAL_T1_DEC_ADAPT=0 unless set), so the GPU is always exercised. With -require-gpu (OPJ_USE_METAL builds) it also fails if
 * the GPU did not decode any code-block. With -output-mct-domain it also
 * decodes with OUTPUT_MCT_DOMAIN=YES (CPU and GPU), applies the inverse MCT
 * and DC level shift of openjpeg.h on the host, and requires the result to
 * equal the normal decode.
 *
 * -concurrent N runs N decoder instances in N threads of this process. Each
 * first decodes every input once, untimed (warm-up, reference hashes), then
 * all start together and do -iters rounds over all inputs (each thread
 * starting at a different one); the last round is checked against the
 * reference hashes. Reports the aggregate frames per second.
 * With several -i files and no -concurrent, the mean of the per-file
 * medians is printed as well. -warmup N first does N untimed decodes,
 * cycling through the inputs (with the same settings), so that the adaptive
 * CPU/GPU split of the Metal decoder has settled before measuring.
 *
 * Encoder settings of the synthetic frame: -mode cinema picks the Cinema
 * 2K/4K profile from the width (with -bitrate MBIT the Cinema size limits of
 * that bit rate at 24 fps), irr is 9/7 with compression ratio -rate, rev is
 * 5/3 lossless; -cblk sets the code-block size (irr, rev), -layers the
 * number of quality layers (irr: ratios rate * 2^k), -cblksty the
 * code-block style bits (1 BYPASS, 2 RESET, 4 TERMALL, 8 VSC, 16 PTERM,
 * 32 SEGSYM). Exits non-zero on any mismatch or failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>

#include "opj_includes.h"
#ifdef OPJ_USE_METAL
#include "opj_metal_t1_dec.h"
#endif

#define NUM_COMPS 3
#define MAX_FILES 256

#include "bench_util.h"

typedef struct {
    /* synthetic frame */
    const char* mode;
    int w, h, prec, mct;
    float rate;
    int max_cs_size, max_comp_size;
    int cblk, layers, cblksty;
    int gray_seed, gray_bits;
    /* decoding */
    int threads;
    int iters;
    int reduce;
    int area[4];
    int has_area;
    int metal; /* -1 option not set, 0 METAL_T1=NO, 1 METAL_T1=YES */
    int mct_domain;
} cfg_t;

typedef struct {
    OPJ_BYTE* buf;
    size_t len;
    char name[256];
} cs_t;

/* ---------------------------------------------------------------------- */
/* Encoding of the synthetic frame                                         */

static int encode_synthetic(const cfg_t* cfg, cs_t* out)
{
    const size_t n = (size_t)cfg->w * (size_t)cfg->h;
    opj_image_cmptparm_t cmptparm[NUM_COMPS];
    opj_cparameters_t params;
    opj_codec_t* codec;
    opj_stream_t* stream;
    opj_image_t* image;
    mem_sink_t sink;
    OPJ_BOOL ok;
    int c, l;

    memset(cmptparm, 0, sizeof(cmptparm));
    for (c = 0; c < NUM_COMPS; c++) {
        cmptparm[c].dx = 1;
        cmptparm[c].dy = 1;
        cmptparm[c].w = (OPJ_UINT32)cfg->w;
        cmptparm[c].h = (OPJ_UINT32)cfg->h;
        cmptparm[c].prec = (OPJ_UINT32)cfg->prec;
        cmptparm[c].sgnd = 0;
    }
    image = opj_image_create(NUM_COMPS, cmptparm, OPJ_CLRSPC_SRGB);
    if (!image) {
        fprintf(stderr, "image creation failed\n");
        return 0;
    }
    image->x0 = 0;
    image->y0 = 0;
    image->x1 = (OPJ_UINT32)cfg->w;
    image->y1 = (OPJ_UINT32)cfg->h;
    for (c = 0; c < NUM_COMPS; c++) {
        if (cfg->gray_seed >= 0) {
            fill_gray_noise(image->comps[c].data, cfg->w, cfg->h, cfg->prec,
                            (unsigned)cfg->gray_seed, cfg->gray_bits);
        } else {
            fill_synthetic(image->comps[c].data, cfg->w, cfg->h, c, cfg->prec);
        }
    }
    (void)n;

    opj_set_default_encoder_parameters(&params);
    params.tcp_numlayers = 1;
    params.cp_disto_alloc = 1;
    params.tcp_rates[0] = 0;
    params.tcp_mct = (char)(cfg->mct ? 1 : 0);
    if (strcmp(cfg->mode, "cinema") == 0) {
        params.rsiz = (cfg->w > 2048) ? OPJ_PROFILE_CINEMA_4K :
                      OPJ_PROFILE_CINEMA_2K;
        params.max_comp_size = cfg->max_comp_size;
        params.max_cs_size = cfg->max_cs_size;
        if (cfg->w > 2048) {
            params.numresolution = 7;
        }
    } else if (strcmp(cfg->mode, "irr") == 0 || strcmp(cfg->mode, "rev") == 0) {
        const int irr = strcmp(cfg->mode, "irr") == 0;
        params.irreversible = irr;
        params.numresolution = 6;
        params.cblockw_init = cfg->cblk;
        params.cblockh_init = cfg->cblk;
        params.mode = cfg->cblksty;
        if (cfg->layers > 1) {
            params.tcp_numlayers = cfg->layers;
            for (l = 0; l < cfg->layers; l++) {
                /* decreasing compression ratios, the last one lossless for */
                /* rev (0) */
                params.tcp_rates[l] = (float)(cfg->rate *
                                              (double)(1 << (cfg->layers - 1 - l)));
            }
            if (!irr) {
                params.tcp_rates[cfg->layers - 1] = 0;
            }
        } else if (irr) {
            params.tcp_rates[0] = cfg->rate;
        }
    } else {
        fprintf(stderr, "unknown mode %s\n", cfg->mode);
        opj_image_destroy(image);
        return 0;
    }

    memset(&sink, 0, sizeof(sink));
    codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_info_handler(codec, quiet_callback, NULL);
    opj_set_warning_handler(codec, quiet_callback, NULL);
    opj_set_error_handler(codec, error_callback, NULL);
    ok = opj_setup_encoder(codec, &params, image);
    if (ok) {
        const char* opts[] = { "METAL_T1=NO", NULL };
        ok = opj_encoder_set_extra_options(codec, opts);
    }
    if (ok) {
        /* the encoder is not what is measured: use all the cores */
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpu > 1) {
            opj_codec_set_threads(codec, (int)ncpu);
        }
    }
    stream = opj_stream_create(1 << 20, OPJ_FALSE);
    opj_stream_set_write_function(stream, sink_write);
    opj_stream_set_skip_function(stream, sink_skip);
    opj_stream_set_seek_function(stream, sink_seek);
    opj_stream_set_user_data(stream, &sink, NULL);
    ok = ok && opj_start_compress(codec, image, stream) &&
         opj_encode(codec, stream) && opj_end_compress(codec, stream);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);
    if (!ok) {
        fprintf(stderr, "encoding the synthetic frame failed\n");
        opj_free(sink.buf);
        return 0;
    }
    out->buf = sink.buf;
    out->len = sink.len;
    snprintf(out->name, sizeof(out->name), "synthetic %s %dx%d mct %d%s",
             cfg->mode, cfg->w, cfg->h, cfg->mct,
             cfg->layers > 1 ? " (layers)" : "");
    return 1;
}

static int load_file(const char* path, cs_t* out)
{
    FILE* f = fopen(path, "rb");
    long len;
    if (!f) {
        perror(path);
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) <= 0 ||
            fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot size %s\n", path);
        fclose(f);
        return 0;
    }
    out->buf = (OPJ_BYTE*)opj_malloc((size_t)len);
    if (!out->buf || fread(out->buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "cannot read %s\n", path);
        fclose(f);
        return 0;
    }
    fclose(f);
    out->len = (size_t)len;
    snprintf(out->name, sizeof(out->name), "%s", path);
    return 1;
}

/* ---------------------------------------------------------------------- */
/* Decoding                                                                 */

typedef struct {
    int metal;       /* -1 not set, 0 NO, 1 YES */
    int threads;
    int reduce;
    int has_area;
    int area[4];
    int mct_domain;
} dec_opts_t;

/* Decodes cs; returns the image (NULL on failure) and its wall time */
static opj_image_t* decode_cs(const cs_t* cs, const dec_opts_t* o,
                              double* ms, int* used_mct)
{
    opj_dparameters_t dparams;
    opj_codec_t* codec;
    opj_stream_t* stream;
    opj_image_t* image = NULL;
    mem_src_t src;
    OPJ_BOOL ok;
    double t0;
    const char* opts[3] = { NULL, NULL, NULL };
    int no = 0;

    src.buf = cs->buf;
    src.len = cs->len;
    src.pos = 0;
    opj_set_default_decoder_parameters(&dparams);
    dparams.cp_reduce = (OPJ_UINT32)o->reduce;
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

    if (o->metal >= 0) {
        opts[no++] = o->metal ? "METAL_T1=YES" : "METAL_T1=NO";
    }
    if (o->mct_domain) {
        opts[no++] = "OUTPUT_MCT_DOMAIN=YES";
    }
    ok = opj_setup_decoder(codec, &dparams);
    if (ok && no) {
        ok = opj_decoder_set_extra_options(codec, opts);
        if (!ok) {
            fprintf(stderr, "opj_decoder_set_extra_options failed\n");
        }
    }
    if (ok && o->threads > 1) {
        ok = opj_codec_set_threads(codec, o->threads);
    }
    t0 = now_ms();
    ok = ok && opj_read_header(stream, codec, &image);
    if (ok && used_mct) {
        opj_codestream_info_v2_t* info = opj_get_cstr_info(codec);
        /* mct | qmfbid << 8 of the default tile */
        *used_mct = 0;
        if (info) {
            *used_mct = (int)info->m_default_tile_info.mct;
            if (info->m_default_tile_info.tccp_info) {
                *used_mct |= (int)info->m_default_tile_info.tccp_info[0].qmfbid << 8;
            }
        }
        opj_destroy_cstr_info(&info);
    }
    if (ok && o->has_area) {
        ok = opj_set_decode_area(codec, image, o->area[0], o->area[1],
                                 o->area[2], o->area[3]);
    }
    ok = ok && opj_decode(codec, stream, image) &&
         opj_end_decompress(codec, stream);
    if (ms) {
        *ms = now_ms() - t0;
    }
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    if (!ok) {
        opj_image_destroy(image);
        return NULL;
    }
    return image;
}

static OPJ_UINT64 image_hash(const opj_image_t* im)
{
    OPJ_UINT64 h = 0;
    OPJ_UINT32 c;
    for (c = 0; c < im->numcomps; c++) {
        const opj_image_comp_t* comp = &im->comps[c];
        if (comp->data) {
            h = h * 31u + fnv1a64((const OPJ_BYTE*)comp->data,
                                  (size_t)comp->w * comp->h * sizeof(OPJ_INT32));
        }
    }
    return h;
}

/* 1 if both images have the same geometry and bit-identical planes */
static int images_equal(const opj_image_t* a, const opj_image_t* b,
                        const char* what)
{
    OPJ_UINT32 c;
    if (a->numcomps != b->numcomps) {
        fprintf(stderr, "  %s: numcomps %u vs %u\n", what, a->numcomps,
                b->numcomps);
        return 0;
    }
    for (c = 0; c < a->numcomps; c++) {
        const opj_image_comp_t* ca = &a->comps[c];
        const opj_image_comp_t* cb = &b->comps[c];
        size_t n, i, bad = 0, first = 0;
        if (ca->w != cb->w || ca->h != cb->h || ca->x0 != cb->x0 ||
                ca->y0 != cb->y0 || ca->prec != cb->prec) {
            fprintf(stderr, "  %s: component %u geometry differs\n", what, c);
            return 0;
        }
        if (!ca->data || !cb->data) {
            if (ca->data != cb->data) {
                fprintf(stderr, "  %s: component %u decoded only once\n", what, c);
                return 0;
            }
            continue;
        }
        n = (size_t)ca->w * ca->h;
        if (memcmp(ca->data, cb->data, n * sizeof(OPJ_INT32)) == 0) {
            continue;
        }
        for (i = 0; i < n; i++) {
            if (ca->data[i] != cb->data[i]) {
                if (bad == 0) {
                    first = i;
                }
                bad++;
            }
        }
        fprintf(stderr, "  %s: component %u: %zu samples differ, first at "
                "(%zu, %zu): %d vs %d\n", what, c, bad, first % ca->w,
                first / ca->w, ca->data[first], cb->data[first]);
        return 0;
    }
    return 1;
}

/* Applies the inverse MCT and DC level shift documented for
 * OUTPUT_MCT_DOMAIN=YES in openjpeg.h, in place, with the same operations
 * in the same order as opj_mct_decode(), opj_mct_decode_real() and
 * opj_tcd_dc_level_shift_decode(). */
static void from_mct_domain(opj_image_t* im, int reversible, int mct)
{
    OPJ_UINT32 c;
    size_t i, n = (size_t)im->comps[0].w * im->comps[0].h;
    if (mct && im->numcomps >= 3) {
        if (reversible) {
            OPJ_INT32* c0 = im->comps[0].data;
            OPJ_INT32* c1 = im->comps[1].data;
            OPJ_INT32* c2 = im->comps[2].data;
            for (i = 0; i < n; i++) {
                OPJ_INT32 y = c0[i];
                OPJ_INT32 u = c1[i];
                OPJ_INT32 v = c2[i];
                OPJ_INT32 g = y - ((u + v) >> 2);
                OPJ_INT32 r = v + g;
                OPJ_INT32 b = u + g;
                c0[i] = r;
                c1[i] = g;
                c2[i] = b;
            }
        } else {
            OPJ_FLOAT32* c0 = (OPJ_FLOAT32*)im->comps[0].data;
            OPJ_FLOAT32* c1 = (OPJ_FLOAT32*)im->comps[1].data;
            OPJ_FLOAT32* c2 = (OPJ_FLOAT32*)im->comps[2].data;
            for (i = 0; i < n; i++) {
                OPJ_FLOAT32 y = c0[i];
                OPJ_FLOAT32 u = c1[i];
                OPJ_FLOAT32 v = c2[i];
                OPJ_FLOAT32 r = y + (v * 1.402f);
                OPJ_FLOAT32 g = y - (u * 0.34413f) - (v * (0.71414f));
                OPJ_FLOAT32 b = y + (u * 1.772f);
                c0[i] = r;
                c1[i] = g;
                c2[i] = b;
            }
        }
    }
    for (c = 0; c < im->numcomps; c++) {
        opj_image_comp_t* comp = &im->comps[c];
        const OPJ_INT32 shift = comp->sgnd ? 0 : 1 << (comp->prec - 1);
        OPJ_INT32 lo, hi;
        n = (size_t)comp->w * comp->h;
        if (comp->sgnd) {
            lo = -(1 << (comp->prec - 1));
            hi = (1 << (comp->prec - 1)) - 1;
        } else {
            lo = 0;
            hi = (OPJ_INT32)((1U << comp->prec) - 1);
        }
        for (i = 0; i < n; i++) {
            if (reversible) {
                comp->data[i] = opj_int_clamp(comp->data[i] + shift, lo, hi);
            } else {
                OPJ_FLOAT32 f;
                memcpy(&f, &comp->data[i], sizeof(f));
                if (f > (OPJ_FLOAT32)INT_MAX) {
                    comp->data[i] = hi;
                } else if (f < INT_MIN) {
                    comp->data[i] = lo;
                } else {
                    comp->data[i] = (OPJ_INT32)opj_int64_clamp(
                                        (OPJ_INT64)opj_lrintf(f) + shift, lo, hi);
                }
            }
        }
    }
}

static OPJ_UINT64 gpu_count(void)
{
#ifdef OPJ_USE_METAL
    return opj_metal_t1_dec_gpu_cblk_count();
#else
    return 0;
#endif
}

/* Number of resolutions of the codestream (from the default COD) */
static int cs_numres(const cs_t* cs)
{
    dec_opts_t o;
    opj_dparameters_t dparams;
    opj_codec_t* codec;
    opj_stream_t* stream;
    opj_image_t* image = NULL;
    mem_src_t src;
    int numres = 1;
    memset(&o, 0, sizeof(o));
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
    if (opj_setup_decoder(codec, &dparams) &&
            opj_read_header(stream, codec, &image)) {
        opj_codestream_info_v2_t* info = opj_get_cstr_info(codec);
        if (info && info->m_default_tile_info.tccp_info) {
            numres = (int)info->m_default_tile_info.tccp_info[0].numresolutions;
        }
        opj_destroy_cstr_info(&info);
    }
    opj_image_destroy(image);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    return numres;
}

static int verify_cs(const cs_t* cs, const cfg_t* cfg, int require_gpu)
{
    const int numres = cs_numres(cs);
    int reduce, ti, ai, ok = 1, ncases = 0;
    const int thread_counts[2] = { 1, cfg->threads > 1 ? cfg->threads : 4 };
    OPJ_UINT64 gpu_before = gpu_count();

    printf("%s (%zu bytes, %d resolutions)\n", cs->name, cs->len, numres);
    for (reduce = 0; reduce <= 2 && reduce < numres; reduce++) {
        for (ti = 0; ti < 2; ti++) {
            for (ai = 0; ai < 2; ai++) {
                dec_opts_t o;
                opj_image_t *a, *b;
                int used_mct = 0;
                char what[128];
                memset(&o, 0, sizeof(o));
                o.threads = thread_counts[ti];
                o.reduce = reduce;
                if (ai == 1) {
                    /* a window not aligned on code-blocks, in full-resolution */
                    /* coordinates */
                    opj_image_t* full;
                    o.metal = 0;
                    full = decode_cs(cs, &o, NULL, NULL);
                    if (!full) {
                        fprintf(stderr, "  CPU decode failed\n");
                        return 0;
                    }
                    o.has_area = 1;
                    o.area[0] = (int)(full->x0 + (full->x1 - full->x0) / 5 + 3);
                    o.area[1] = (int)(full->y0 + (full->y1 - full->y0) / 4 + 5);
                    o.area[2] = (int)(full->x0 + (full->x1 - full->x0) * 3 / 4 + 1);
                    o.area[3] = (int)(full->y0 + (full->y1 - full->y0) * 2 / 3 + 7);
                    opj_image_destroy(full);
                }
                snprintf(what, sizeof(what), "reduce %d, %d thread%s, %s", reduce,
                         o.threads, o.threads > 1 ? "s" : "",
                         ai ? "window" : "whole image");
                /* The reference also uses the original scalar DC level */
                /* shift, so the NEON one (dc_shift.h) is checked too */
                o.metal = 0;
                setenv("OPJ_DEC_SCALAR_DC_SHIFT", "1", 1);
                a = decode_cs(cs, &o, NULL, &used_mct);
                unsetenv("OPJ_DEC_SCALAR_DC_SHIFT");
                o.metal = 1;
                b = decode_cs(cs, &o, NULL, NULL);
                ncases++;
                if (!a || !b) {
                    fprintf(stderr, "  %s: decode failed (cpu %s, gpu %s)\n", what,
                            a ? "ok" : "failed", b ? "ok" : "failed");
                    ok = 0;
                } else if (!images_equal(a, b, what)) {
                    ok = 0;
                } else if (cfg->mct_domain) {
                    /* OUTPUT_MCT_DOMAIN=YES, CPU and GPU, back to samples */
                    int m;
                    const int reversible = (used_mct >> 8) == 1;
                    for (m = 0; m < 2 && ok; m++) {
                        opj_image_t* d;
                        char what2[160];
                        o.metal = m;
                        o.mct_domain = 1;
                        d = decode_cs(cs, &o, NULL, NULL);
                        o.mct_domain = 0;
                        snprintf(what2, sizeof(what2), "%s, OUTPUT_MCT_DOMAIN (%s)",
                                 what, m ? "gpu" : "cpu");
                        if (!d) {
                            fprintf(stderr, "  %s: decode failed\n", what2);
                            ok = 0;
                            break;
                        }
                        from_mct_domain(d, reversible, (used_mct & 0xff) == 1);
                        if (!images_equal(a, d, what2)) {
                            ok = 0;
                        }
                        opj_image_destroy(d);
                    }
                }
                opj_image_destroy(a);
                opj_image_destroy(b);
            }
        }
    }
    {
        const OPJ_UINT64 gpu = gpu_count() - gpu_before;
        printf("  %d cases %s, %llu code-blocks decoded on the GPU\n", ncases,
               ok ? "identical" : "FAILED", (unsigned long long)gpu);
        if (ok && require_gpu && gpu == 0) {
            fprintf(stderr, "  -require-gpu: the GPU was not used\n");
            ok = 0;
        }
    }
    return ok;
}

static int bench_cs(const cs_t* cs, const cfg_t* cfg, double* median)
{
    dec_opts_t o;
    double* times = (double*)opj_malloc(sizeof(double) * (size_t)cfg->iters);
    OPJ_UINT64 hash = 0, gpu_before = gpu_count();
    double mean = 0.0;
    int i;
    memset(&o, 0, sizeof(o));
    o.metal = cfg->metal;
    o.threads = cfg->threads;
    o.reduce = cfg->reduce;
    o.has_area = cfg->has_area;
    memcpy(o.area, cfg->area, sizeof(o.area));
    o.mct_domain = cfg->mct_domain;
    for (i = 0; i < cfg->iters; i++) {
        opj_image_t* im = decode_cs(cs, &o, &times[i], NULL);
        if (!im) {
            fprintf(stderr, "%s: decode failed\n", cs->name);
            opj_free(times);
            return 0;
        }
        hash = image_hash(im);
        opj_image_destroy(im);
    }
    for (i = 0; i < cfg->iters; i++) {
        mean += times[i] / cfg->iters;
    }
    qsort(times, (size_t)cfg->iters, sizeof(double), compare_double);
    *median = times[cfg->iters / 2];
    printf("%s: %zu bytes, decode median %.1f ms, mean %.1f ms, min %.1f ms, "
           "hash %016llx, %llu gpu code-blocks per decode\n", cs->name, cs->len,
           times[cfg->iters / 2], mean, times[0], (unsigned long long)hash,
           (unsigned long long)((gpu_count() - gpu_before) / (OPJ_UINT64)cfg->iters));
#ifdef OPJ_USE_METAL
    if (cfg->metal == 1) {
        OPJ_UINT32 nb = 0;
        OPJ_SIZE_T bytes = opj_metal_t1_dec_pool_bytes(&nb);
        printf("metal batch pool: %u batches, %.1f MB (%.1f MB per batch)\n",
               nb, (double)bytes / 1e6, nb ? (double)bytes / 1e6 / nb : 0.0);
    }
#endif
    opj_free(times);
    return 1;
}

typedef struct {
    const cs_t* css;
    int ncs;
    int offset;
    dec_opts_t o;
    int iters;
    OPJ_UINT64* hashes; /* per file, from the warm-up decode */
    int ok;
} thread_arg_t;

/* Start gate: the threads warm up (untimed), then start together */
static pthread_mutex_t g_gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_gate_cond = PTHREAD_COND_INITIALIZER;
static int g_gate_ready = 0;
static int g_gate_go = 0;

static void* bench_thread(void* p)
{
    thread_arg_t* a = (thread_arg_t*)p;
    int i, k;
    a->ok = 1;
    /* Warm-up: decode every file once (pays for the Metal shader compile */
    /* and first-touch allocations) and record its hash */
    for (k = 0; k < a->ncs && a->ok; k++) {
        opj_image_t* im = decode_cs(&a->css[k], &a->o, NULL, NULL);
        if (!im) {
            a->ok = 0;
            break;
        }
        a->hashes[k] = image_hash(im);
        opj_image_destroy(im);
    }
    pthread_mutex_lock(&g_gate_mutex);
    g_gate_ready++;
    pthread_cond_broadcast(&g_gate_cond);
    while (!g_gate_go) {
        pthread_cond_wait(&g_gate_cond, &g_gate_mutex);
    }
    pthread_mutex_unlock(&g_gate_mutex);
    /* Timed: only the last round is hashed, so hashing stays out of the */
    /* measured throughput as much as possible */
    for (i = 0; i < a->iters && a->ok; i++) {
        for (k = 0; k < a->ncs; k++) {
            const int f = (a->offset + k) % a->ncs;
            opj_image_t* im = decode_cs(&a->css[f], &a->o, NULL, NULL);
            if (!im) {
                a->ok = 0;
                break;
            }
            if (i == a->iters - 1 && image_hash(im) != a->hashes[f]) {
                a->ok = 0;
            }
            opj_image_destroy(im);
        }
    }
    return NULL;
}

/* n decoders in n threads, each cycling through all files (starting at a */
/* different one), iters rounds each */
static int concurrent_run(const cs_t* css, int ncs, const cfg_t* cfg, int n)
{
    pthread_t* th = (pthread_t*)opj_malloc(sizeof(pthread_t) * (size_t)n);
    thread_arg_t* args = (thread_arg_t*)opj_calloc((size_t)n, sizeof(*args));
    double t0, dt;
    int i, k, ok = 1;
    g_gate_ready = 0;
    g_gate_go = 0;
    for (i = 0; i < n; i++) {
        args[i].css = css;
        args[i].ncs = ncs;
        args[i].offset = i % ncs;
        args[i].o.metal = cfg->metal;
        args[i].o.threads = cfg->threads;
        args[i].o.reduce = cfg->reduce;
        args[i].o.mct_domain = cfg->mct_domain;
        args[i].iters = cfg->iters;
        args[i].hashes = (OPJ_UINT64*)opj_calloc((size_t)ncs, sizeof(OPJ_UINT64));
        pthread_create(&th[i], NULL, bench_thread, &args[i]);
    }
    pthread_mutex_lock(&g_gate_mutex);
    while (g_gate_ready < n) {
        pthread_cond_wait(&g_gate_cond, &g_gate_mutex);
    }
    t0 = now_ms();
    g_gate_go = 1;
    pthread_cond_broadcast(&g_gate_cond);
    pthread_mutex_unlock(&g_gate_mutex);
    for (i = 0; i < n; i++) {
        pthread_join(th[i], NULL);
    }
    dt = now_ms() - t0;
    for (i = 0; i < n; i++) {
        if (!args[i].ok) {
            ok = 0;
        }
        for (k = 0; k < ncs; k++) {
            if (args[i].hashes[k] != args[0].hashes[k]) {
                ok = 0;
            }
        }
    }
    printf("%s%s: %d decoders x %d frames in %.1f ms: %.1f fps, outputs %s\n",
           css[0].name, ncs > 1 ? " (and others)" : "", n, cfg->iters * ncs, dt,
           1e3 * n * cfg->iters * ncs / dt, ok ? "identical" : "DIFFER");
    for (i = 0; i < n; i++) {
        opj_free(args[i].hashes);
    }
    opj_free(th);
    opj_free(args);
    return ok;
}

int main(int argc, char** argv)
{
    cfg_t cfg;
    cs_t css[MAX_FILES];
    const char* files[MAX_FILES];
    int nfiles = 0, ncs = 0;
    int verify = 0, require_gpu = 0, concurrent = 0, warmup = 0;
    const char* outfile = NULL;
    int i, ok = 1;

    memset(&cfg, 0, sizeof(cfg));
    cfg.mode = "cinema";
    cfg.w = 4096;
    cfg.h = 2160;
    cfg.prec = 12;
    cfg.mct = 1;
    cfg.rate = 20.0f;
    cfg.max_cs_size = OPJ_CINEMA_24_CS;
    cfg.max_comp_size = OPJ_CINEMA_24_COMP;
    cfg.cblk = 32;
    cfg.layers = 1;
    cfg.gray_seed = -1;
    cfg.gray_bits = 10;
    cfg.threads = 1;
    cfg.iters = 5;
    cfg.metal = -1;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (strcmp(argv[i], "-mct") == 0 && i + 1 < argc) {
            cfg.mct = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "-size") == 0 && i + 2 < argc) {
            cfg.w = atoi(argv[++i]);
            cfg.h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-prec") == 0 && i + 1 < argc) {
            cfg.prec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-rate") == 0 && i + 1 < argc) {
            cfg.rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-bitrate") == 0 && i + 1 < argc) {
            const long long datarate = (long long)(atof(argv[++i]) * 1e6);
            cfg.max_cs_size = (int)(datarate / 8 / 24);
            cfg.max_comp_size = (int)(cfg.max_cs_size / 1.25);
        } else if (strcmp(argv[i], "-cblk") == 0 && i + 1 < argc) {
            cfg.cblk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-layers") == 0 && i + 1 < argc) {
            cfg.layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-cblksty") == 0 && i + 1 < argc) {
            cfg.cblksty = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-gray-noise") == 0 && i + 2 < argc) {
            cfg.gray_seed = atoi(argv[++i]);
            cfg.gray_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outfile = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-' && nfiles < MAX_FILES) {
                files[nfiles++] = argv[++i];
            }
        } else if (strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            cfg.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-iters") == 0 && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-reduce") == 0 && i + 1 < argc) {
            cfg.reduce = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-area") == 0 && i + 4 < argc) {
            int k;
            for (k = 0; k < 4; k++) {
                cfg.area[k] = atoi(argv[++i]);
            }
            cfg.has_area = 1;
        } else if (strcmp(argv[i], "-metal") == 0) {
            cfg.metal = 1;
        } else if (strcmp(argv[i], "-cpu") == 0) {
            cfg.metal = 0;
        } else if (strcmp(argv[i], "-verify") == 0) {
            verify = 1;
        } else if (strcmp(argv[i], "-require-gpu") == 0) {
            require_gpu = 1;
        } else if (strcmp(argv[i], "-concurrent") == 0 && i + 1 < argc) {
            concurrent = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-output-mct-domain") == 0) {
            cfg.mct_domain = 1;
        } else {
            fprintf(stderr,
                    "usage: bench_decode [-mode cinema|irr|rev] [-mct 0|1] "
                    "[-size W H] [-prec P] [-rate R] [-bitrate MBIT] [-cblk N] "
                    "[-layers N] [-cblksty N] [-gray-noise SEED BITS] "
                    "[-o file.j2k] [-i file.j2c ...] [-threads N] [-iters N] "
                    "[-reduce R] [-area X0 Y0 X1 Y1] [-metal | -cpu] [-verify] "
                    "[-require-gpu] [-concurrent N] [-warmup N] "
                    "[-output-mct-domain]\n");
            return 1;
        }
    }
    if (cfg.iters < 1) {
        cfg.iters = 1;
    }
    if (cfg.layers < 1 || cfg.layers > 10 || cfg.w < 1 || cfg.h < 1 ||
            cfg.prec < 1 || cfg.prec > 16) {
        fprintf(stderr, "invalid settings\n");
        return 1;
    }

    if (verify && !getenv("OPJ_METAL_T1_DEC_ADAPT")) {
        /* The adaptive CPU/GPU split may settle on "CPU only": verify the */
        /* GPU path every time instead (the fixed formula's split) */
        setenv("OPJ_METAL_T1_DEC_ADAPT", "0", 1);
    }
    if (nfiles) {
        for (i = 0; i < nfiles; i++) {
            if (!load_file(files[i], &css[ncs])) {
                return 1;
            }
            ncs++;
        }
    } else {
        if (!encode_synthetic(&cfg, &css[0])) {
            return 1;
        }
        ncs = 1;
        if (outfile) {
            FILE* f = fopen(outfile, "wb");
            if (!f || fwrite(css[0].buf, 1, css[0].len, f) != css[0].len) {
                perror(outfile);
                return 1;
            }
            fclose(f);
        }
    }

    if (warmup > 0 && !verify) {
        /* untimed decodes first, cycling through the inputs: lets the */
        /* adaptive CPU/GPU split settle before measuring */
        dec_opts_t o;
        memset(&o, 0, sizeof(o));
        o.metal = cfg.metal;
        o.threads = cfg.threads;
        o.reduce = cfg.reduce;
        o.mct_domain = cfg.mct_domain;
        for (i = 0; i < warmup; i++) {
            opj_image_t* im = decode_cs(&css[i % ncs], &o, NULL, NULL);
            if (im) {
                opj_image_destroy(im);
            }
        }
    }
    if (concurrent > 0 && !verify) {
        ok = concurrent_run(css, ncs, &cfg, concurrent);
    } else {
        double sum = 0.0, median = 0.0;
        for (i = 0; i < ncs; i++) {
            if (verify) {
                ok = verify_cs(&css[i], &cfg, require_gpu) && ok;
            } else {
                ok = bench_cs(&css[i], &cfg, &median) && ok;
                sum += median;
            }
        }
        if (!verify && ncs > 1) {
            printf("%d files: mean of the medians %.2f ms\n", ncs, sum / ncs);
        }
    }
    for (i = 0; i < ncs; i++) {
        opj_free(css[i].buf);
    }
    if (verify) {
        printf(ok ? "PASS\n" : "FAIL\n");
    }
    return ok ? 0 : 1;
}
