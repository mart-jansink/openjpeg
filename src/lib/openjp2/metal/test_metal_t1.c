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
 * test_metal_t1: checks the Metal T1 encoder against OPJ_T1_DUMP dumps of
 * the CPU encoder (format documented in t1.c) and measures throughput.
 *
 * Usage:
 *   test_metal_t1 [-kernel serial|parallel|both] [-batch N] [-reps N]
 *                 [-max N] dump1 [dump2 ...]
 *
 * Each dump is split into batches of -batch code-blocks (default 32768,
 * one 4K frame). Every batch is encoded once as warm-up and then
 * -reps times (default 5) for timing; the output of the last run is
 * compared with the CPU results. Exit code 0 if everything matches.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opj_metal_t1.h"

#define DUMP_TAG 0x4B4C4243U

typedef struct {
    OPJ_UINT32 w, h, orient, qmfbid, cblksty, band_numbps, compno, level;
    OPJ_UINT32 numbps, totalpasses, nbytes;
    double cpu_s;
    const OPJ_INT32* data;
    const OPJ_UINT32* passes; /* 4 per pass: rate, len, term, nmsedec */
    const OPJ_BYTE* bytes;
} rec_t;

typedef struct {
    unsigned long cblks, bad, bad_numbps, bad_passes, bad_rate, bad_len,
             bad_term, bad_nmsedec, bad_bytes, cmp_passes, refused;
    double cpu_s, gpu_total_s, gpu_exec_s, prep_s;
    unsigned long timed_cblks;
    int printed;
} stats_t;

static unsigned char* read_file(const char* path, size_t* size)
{
    FILE* f = fopen(path, "rb");
    unsigned char* buf;
    long n;
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char*)malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

static size_t parse_dump(unsigned char* buf, size_t size, rec_t** precs)
{
    size_t pos = 16, n = 0, cap = 1024;
    rec_t* recs = (rec_t*)malloc(cap * sizeof(rec_t));
    if (size < 16 || memcmp(buf, "OPJT1DMP", 8) != 0) {
        fprintf(stderr, "bad dump header\n");
        exit(2);
    }
    while (pos + 8 <= size) {
        OPJ_UINT32 hdr[13];
        rec_t* r;
        memcpy(hdr, buf + pos, 8);
        if (hdr[0] != DUMP_TAG || pos + 8 + hdr[1] > size) {
            fprintf(stderr, "bad record at offset %zu\n", pos);
            exit(2);
        }
        memcpy(hdr, buf + pos, sizeof(hdr));
        if (n == cap) {
            cap *= 2;
            recs = (rec_t*)realloc(recs, cap * sizeof(rec_t));
        }
        r = &recs[n++];
        r->w = hdr[2];
        r->h = hdr[3];
        r->orient = hdr[4];
        r->qmfbid = hdr[5];
        r->cblksty = hdr[6];
        r->band_numbps = hdr[7];
        r->compno = hdr[8];
        r->level = hdr[9];
        r->numbps = hdr[10];
        r->totalpasses = hdr[11];
        r->nbytes = hdr[12];
        memcpy(&r->cpu_s, buf + pos + 52, 8);
        r->data = (const OPJ_INT32*)(buf + pos + 60);
        r->passes = (const OPJ_UINT32*)(buf + pos + 60 + (size_t)r->w * r->h * 4);
        r->bytes = (const OPJ_BYTE*)(r->passes + 4 * r->totalpasses);
        pos += 8 + hdr[1];
    }
    *precs = recs;
    return n;
}

static void report(stats_t* st, unsigned long idx, const rec_t* r,
                   const char* what, int passno, long expected, long actual)
{
    if (st->printed >= 10) {
        return;
    }
    st->printed++;
    printf("  MISMATCH cblk #%lu (%ux%u orient=%u qmfbid=%u comp=%u level=%u "
           "numbps=%u passes=%u nbytes=%u): %s", idx, r->w, r->h, r->orient,
           r->qmfbid, r->compno, r->level, r->numbps, r->totalpasses, r->nbytes,
           what);
    if (passno >= 0) {
        printf(strncmp(what, "byte", 4) == 0 ? " %d" : " pass %d", passno);
    }
    printf(" expected %ld got %ld\n", expected, actual);
}

static void compare(stats_t* st, unsigned long base, const rec_t* r,
                    const opj_metal_t1_cblk_out* o, OPJ_UINT32 n)
{
    OPJ_UINT32 i, p;
    for (i = 0; i < n; ++i) {
        const rec_t* c = &r[i];
        const opj_metal_t1_cblk_out* g = &o[i];
        int bad = 0;
        st->cblks++;
        if (g->status != 0) {
            /* handed back to the CPU by design; not a mismatch */
            st->refused++;
            continue;
        }
        if (c->numbps != g->numbps) {
            st->bad_numbps++;
            bad = 1;
            report(st, base + i, c, "numbps", -1, (long)c->numbps, (long)g->numbps);
        }
        if (c->totalpasses != g->totalpasses) {
            st->bad_passes++;
            st->bad++;
            report(st, base + i, c, "totalpasses", -1, c->totalpasses,
                   g->totalpasses);
            continue;
        }
        for (p = 0; p < c->totalpasses; ++p) {
            const OPJ_UINT32* e = c->passes + 4 * p;
            const opj_metal_t1_pass* a = &g->passes[p];
            st->cmp_passes++;
            if (e[0] != a->rate) {
                st->bad_rate++;
                bad = 1;
                report(st, base + i, c, "rate", (int)p, e[0], a->rate);
            }
            if (e[1] != a->len) {
                st->bad_len++;
                bad = 1;
                report(st, base + i, c, "len", (int)p, e[1], a->len);
            }
            if (e[2] != (p + 1 == g->totalpasses ? 1u : 0u)) {
                st->bad_term++;
                bad = 1;
                report(st, base + i, c, "term", (int)p, e[2], !e[2]);
            }
            if ((OPJ_INT32)e[3] != a->nmsedec) {
                st->bad_nmsedec++;
                bad = 1;
                report(st, base + i, c, "nmsedec", (int)p, (OPJ_INT32)e[3],
                       a->nmsedec);
            }
        }
        if (c->nbytes != g->nbytes) {
            st->bad_bytes++;
            bad = 1;
            report(st, base + i, c, "nbytes", -1, c->nbytes, g->nbytes);
        } else if (memcmp(c->bytes, g->data, c->nbytes) != 0) {
            OPJ_UINT32 k = 0;
            while (c->bytes[k] == g->data[k]) {
                k++;
            }
            st->bad_bytes++;
            bad = 1;
            report(st, base + i, c, "byte at offset", (int)k, c->bytes[k],
                   g->data[k]);
        }
        if (bad) {
            st->bad++;
        }
    }
}

static int run_file(opj_metal_t1_t* ctx, opj_metal_t1_kernel_t kernel,
                    const char* path, OPJ_UINT32 batch, int reps,
                    unsigned long maxc, stats_t* st)
{
    size_t size, n, b;
    unsigned char* buf = read_file(path, &size);
    rec_t* recs;
    opj_metal_t1_cblk_in* in;
    opj_metal_t1_cblk_out* out;
    if (!buf) {
        fprintf(stderr, "cannot read %s\n", path);
        return 0;
    }
    n = parse_dump(buf, size, &recs);
    if (maxc && n > maxc) {
        n = maxc;
    }
    in = (opj_metal_t1_cblk_in*)malloc(n * sizeof(*in));
    out = (opj_metal_t1_cblk_out*)malloc(n * sizeof(*out));
    for (b = 0; b < n; ++b) {
        in[b].data = recs[b].data;
        in[b].w = recs[b].w;
        in[b].h = recs[b].h;
        in[b].orient = recs[b].orient;
        in[b].qmfbid = recs[b].qmfbid;
        in[b].cblksty = recs[b].cblksty;
        in[b].band_numbps = recs[b].band_numbps;
        st->cpu_s += recs[b].cpu_s;
    }
    opj_metal_t1_set_kernel(ctx, kernel);
    for (b = 0; b < n; b += batch) {
        OPJ_UINT32 cnt = (OPJ_UINT32)((n - b) < batch ? (n - b) : batch);
        int rep;
        if (!opj_metal_t1_encode_batch(ctx, in + b, out + b, cnt)) {
            fprintf(stderr, "encode_batch failed\n");
            return 0;
        }
        for (rep = 0; rep < reps; ++rep) {
            double prep, gpu, total;
            if (!opj_metal_t1_encode_batch(ctx, in + b, out + b, cnt)) {
                fprintf(stderr, "encode_batch failed\n");
                return 0;
            }
            opj_metal_t1_last_timing(ctx, &prep, &gpu, &total);
            st->prep_s += prep;
            st->gpu_exec_s += gpu;
            st->gpu_total_s += total;
            st->timed_cblks += cnt;
        }
        if (kernel == OPJ_METAL_T1_KERNEL_PARALLEL) {
            OPJ_UINT32 pool, fb;
            opj_metal_t1_last_stats(ctx, &pool, &fb);
            printf("  batch %zu: %u cblks, symbol pool demand %.1f MB (%.0f B/cblk), "
                   "%u GPU retries\n", b / batch, cnt, pool / 1e6,
                   (double)pool / cnt, fb);
        }
        compare(st, (unsigned long)b, recs + b, out + b, cnt);
    }
    free(in);
    free(out);
    free(recs);
    free(buf);
    return 1;
}

int main(int argc, char** argv)
{
    OPJ_UINT32 batch = 32768;
    int reps = 5, i, k, nk = 0, failed = 0;
    unsigned long maxc = 0;
    opj_metal_t1_kernel_t kernels[2];
    const char* kname[2] = { "serial(a)", "parallel(b)" };
    const char* ksel = "both";
    opj_metal_t1_t* ctx;
    int first_file = argc;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-kernel") == 0 && i + 1 < argc) {
            ksel = argv[++i];
        } else if (strcmp(argv[i], "-batch") == 0 && i + 1 < argc) {
            batch = (OPJ_UINT32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-reps") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-max") == 0 && i + 1 < argc) {
            maxc = strtoul(argv[++i], NULL, 10);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "usage: test_metal_t1 [-kernel serial|parallel|both] "
                    "[-batch N] [-reps N] [-max N] dump...\n");
            return 2;
        } else {
            first_file = i;
            break;
        }
    }
    if (first_file >= argc || batch == 0) {
        fprintf(stderr, "no dump file given\n");
        return 2;
    }
    if (strcmp(ksel, "serial") == 0 || strcmp(ksel, "both") == 0) {
        kernels[nk++] = OPJ_METAL_T1_KERNEL_SERIAL;
    }
    if (strcmp(ksel, "parallel") == 0 || strcmp(ksel, "both") == 0) {
        kernels[nk++] = OPJ_METAL_T1_KERNEL_PARALLEL;
    }
    ctx = opj_metal_t1_create();
    if (!ctx) {
        return 1;
    }
    for (k = 0; k < nk; ++k) {
        stats_t tot;
        memset(&tot, 0, sizeof(tot));
        if (!opj_metal_t1_set_kernel(ctx, kernels[k])) {
            printf("kernel %s not available\n", kname[kernels[k]]);
            failed = 1;
            continue;
        }
        printf("=== kernel %s, batch %u, reps %d ===\n", kname[kernels[k]], batch,
               reps);
        for (i = first_file; i < argc; ++i) {
            stats_t st;
            memset(&st, 0, sizeof(st));
            if (!run_file(ctx, kernels[k], argv[i], batch, reps, maxc, &st)) {
                failed = 1;
                continue;
            }
            printf("%s: %lu cblks, %lu passes, mismatching cblks %lu "
                   "(numbps %lu, passes %lu, rate %lu, len %lu, term %lu, "
                   "nmsedec %lu, bytes %lu), handed back to CPU %lu\n", argv[i],
                   st.cblks, st.cmp_passes,
                   st.bad, st.bad_numbps, st.bad_passes, st.bad_rate, st.bad_len,
                   st.bad_term, st.bad_nmsedec, st.bad_bytes, st.refused);
            if (st.timed_cblks) {
                printf("  GPU %.0f cblk/s (wall of encode_batch; GPU exec only "
                       "%.0f cblk/s; host prep %.1f ms/batch-avg)  CPU 1 core "
                       "%.0f cblk/s\n",
                       st.timed_cblks / st.gpu_total_s,
                       st.timed_cblks / st.gpu_exec_s,
                       1e3 * st.prep_s / (st.timed_cblks / (double)
                                          (st.cblks < batch ? st.cblks : batch)),
                       st.cblks / st.cpu_s);
            }
            tot.cblks += st.cblks;
            tot.refused += st.refused;
            tot.bad += st.bad;
            tot.cmp_passes += st.cmp_passes;
            tot.cpu_s += st.cpu_s;
            tot.gpu_total_s += st.gpu_total_s;
            tot.gpu_exec_s += st.gpu_exec_s;
            tot.timed_cblks += st.timed_cblks;
        }
        printf("TOTAL %s: %lu cblks (%lu handed back to CPU), %lu passes, %lu mismatching cblks; "
               "GPU %.0f cblk/s (exec %.0f), %.1f ms per 26k cblks; "
               "CPU %.0f cblk/s\n", kname[kernels[k]], tot.cblks, tot.refused, tot.cmp_passes,
               tot.bad, tot.timed_cblks / tot.gpu_total_s,
               tot.timed_cblks / tot.gpu_exec_s,
               26000.0 * 1e3 * tot.gpu_total_s / tot.timed_cblks,
               tot.cblks / tot.cpu_s);
        if (tot.bad) {
            failed = 1;
        }
    }
    opj_metal_t1_destroy(ctx);
    return failed;
}
