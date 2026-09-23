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
 * Prototype JPEG 2000 Tier-1 (EBCOT) code-block encoder on Apple GPUs
 * through Metal. Output is bit-exact with OpenJPEG's opj_t1_encode_cblk()
 * for code-block style 0 (no BYPASS, RESET, TERMALL, VSC, PTERM, SEGSYM).
 *
 * Plain C API; the implementation is Objective-C (opj_metal_t1.m).
 */

#ifndef OPJ_METAL_T1_H
#define OPJ_METAL_T1_H

#include "openjpeg.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound of the number of passes of a code-block: numbps <= 25 since */
/* the input is scaled by 1 << T1_NMSEDEC_FRACBITS, so 3 * 25 - 2 = 73.    */
#define OPJ_METAL_T1_MAXPASSES 76

/* Maximum code-block area (JPEG 2000 limit) and side supported */
#define OPJ_METAL_T1_MAX_AREA 4096
#define OPJ_METAL_T1_MAX_SIDE 64

typedef struct opj_metal_t1 opj_metal_t1_t;

typedef enum {
    /* Design (a): one GPU thread per code-block */
    OPJ_METAL_T1_KERNEL_SERIAL = 0,
    /* Design (b): one SIMD-group per code-block, bit-plane parallel */
    /* context modelling, serial MQ coder */
    OPJ_METAL_T1_KERNEL_PARALLEL = 1
} opj_metal_t1_kernel_t;

typedef struct opj_metal_t1_cblk_in {
    /* w * h samples in raster (row-major) order, two's complement, already */
    /* quantised and scaled by 1 << T1_NMSEDEC_FRACBITS, i.e. the values    */
    /* opj_t1_cblk_encode_processor() stores into t1->data (but not in its  */
    /* stripe order).                                                       */
    const OPJ_INT32* data;
    OPJ_UINT32 w;
    OPJ_UINT32 h;
    OPJ_UINT32 orient;      /* band->bandno: 0 LL, 1 HL, 2 LH, 3 HH */
    OPJ_UINT32 qmfbid;      /* informative only, T1 is the same for both */
    OPJ_UINT32 cblksty;     /* must be 0 */
    OPJ_UINT32 band_numbps; /* informative only */
} opj_metal_t1_cblk_in;

/* opj_tcd_pass_t::term is 1 for the last pass only (code-block style 0) */
typedef struct opj_metal_t1_pass {
    OPJ_UINT32 rate;    /* opj_tcd_pass_t::rate after the final fix-ups */
    OPJ_UINT32 len;     /* opj_tcd_pass_t::len */
    OPJ_INT32 nmsedec;  /* integer nmsedec of the pass, feed it to */
    /* opj_t1_getwmsedec() */
} opj_metal_t1_pass;

typedef struct opj_metal_t1_cblk_out {
    /* 0: encoded. Otherwise the GPU gave up on this code-block (its coded */
    /* bytes would not fit in its region, which needs more than about 4   */
    /* bytes per sample, or the symbol pool stayed full after retries);   */
    /* the caller must encode it on the CPU. The other fields are then    */
    /* not valid.                                                         */
    OPJ_UINT32 status;
    /* Coded bytes. Points into memory owned by the batch, valid until   */
    /* its next prepare or its release.                                  */
    const OPJ_BYTE* data;
    OPJ_UINT32 data_max;    /* bytes available at data */
    OPJ_UINT32 nbytes;      /* rate of the last pass, 0 if no pass */
    OPJ_UINT32 numbps;      /* opj_tcd_cblk_enc_t::numbps */
    OPJ_UINT32 totalpasses; /* opj_tcd_cblk_enc_t::totalpasses */
    const opj_metal_t1_pass* passes; /* totalpasses entries, same lifetime */
} opj_metal_t1_cblk_out;

/*
 * Threading model. The Metal device, command queue, compiled library and
 * pipelines are process-wide, created lazily and thread-safely on first
 * use. Work is submitted through batches (opj_metal_t1_t): each batch owns
 * its buffers, is used by one thread at a time, and is taken from a
 * process-wide pool bounded by the environment variable
 * OPJ_METAL_T1_MAX_BATCHES (default 3); acquire blocks while all batches
 * are in use. Released batches keep their buffers for reuse. Several
 * threads can have batches in flight at the same time.
 *
 * Memory of a batch. Design (b) keeps a code-block's symbol stream and then
 * its coded bytes inside the code-block's own input region (w*h*4 bytes,
 * at least 256), which is free once the kernel has loaded the samples;
 * only the bit-planes that do not fit go to a shared symbol pool. Per
 * 32x32 code-block: input region 4096 B, symbol pool 1.25 x the average
 * overflow of the previous batch (starts at 3 KB, about 2.8 KB for 12-bit
 * Cinema content, 6.6 KB for 16-bit 5/3), pass records 1.25 x the average
 * pass count (starts at 24 x 12 B), descriptor + result 64 B. Code-blocks
 * larger than 32x32 use design (a) and additionally need the worst-case
 * output (74 + 4*w*h bytes) and flags scratch. Measured for a 3840x2076
 * Cinema 4K frame (24129 code-blocks): 178 MB per batch, 535 MB for the
 * default pool of 3 (it was 421 MB per batch before the in-region scheme).
 * Buffers are allocated at the exact size first, grow with 25 % headroom,
 * and are replaced by an exact one when more than 4 times too big.
 *
 * When the symbol pool or the pass records are full, the kernel restores
 * the code-block's input and the batch runs those code-blocks again with
 * grown buffers (up to 4 rounds). A code-block whose coded bytes do not
 * fit in its region (more than about 4 bytes per sample) is handed back:
 * opj_metal_t1_cblk_out::status != 0, and the caller encodes it on the CPU.
 */

/* Returns OPJ_TRUE if a Metal device is available and the kernels compiled */
/* (the first call initialises the shared state; errors go to stderr).    */
OPJ_BOOL opj_metal_t1_available(void);

/* Takes a batch from the pool, blocking while OPJ_METAL_T1_MAX_BATCHES are */
/* in use. NULL if Metal is not available.                                 */
opj_metal_t1_t* opj_metal_t1_acquire(void);

/* Returns a batch to the pool; its outputs become invalid. */
void opj_metal_t1_release(opj_metal_t1_t* batch);

/* Selects the kernel of the next runs of this batch (default parallel) */
OPJ_BOOL opj_metal_t1_set_kernel(opj_metal_t1_t* batch,
                                 opj_metal_t1_kernel_t kernel);

/* Lays out count code-blocks in the batch buffers. in[i].data is ignored.  */
/* On success data_ptrs[i] points to the place, inside the GPU input buffer */
/* (shared memory), where the caller must write the w*h raster samples of   */
/* code-block i before opj_metal_t1_run(). Returns OPJ_FALSE on an         */
/* unsupported code-block (cblksty != 0, side > 64, area > 4096, empty) or */
/* allocation failure.                                                     */
OPJ_BOOL opj_metal_t1_prepare(opj_metal_t1_t* batch,
                              const opj_metal_t1_cblk_in* in,
                              OPJ_UINT32 count, OPJ_INT32** data_ptrs);

/* Encodes the prepared code-blocks with a single command buffer (plus one */
/* more in the rare symbol pool overflow case) and waits for completion.   */
/* out must have count entries. OPJ_FALSE on a Metal failure (out invalid). */
OPJ_BOOL opj_metal_t1_run(opj_metal_t1_t* batch, opj_metal_t1_cblk_out* out);

/* Convenience for tests: acquire / release under the old names */
opj_metal_t1_t* opj_metal_t1_create(void);
void opj_metal_t1_destroy(opj_metal_t1_t* batch);

/* prepare, copy in[i].data into the input buffer, run */
OPJ_BOOL opj_metal_t1_encode_batch(opj_metal_t1_t* batch,
                                   const opj_metal_t1_cblk_in* in,
                                   opj_metal_t1_cblk_out* out,
                                   OPJ_UINT32 count);

/* Timing of the last run of the batch, in seconds: host preparation      */
/* (layout, plus the input copy for encode_batch), GPU execution          */
/* (GPUStartTime..GPUEndTime) and wall time of prepare(+copy)+run.        */
void opj_metal_t1_last_timing(const opj_metal_t1_t* batch, double* prep_s,
                              double* gpu_s, double* total_s);

/* Statistics of the last run with design (b): bytes of the symbol pool    */
/* used, number of code-blocks re-encoded with design (a) because the pool */
/* was full.                                                              */
void opj_metal_t1_last_stats(const opj_metal_t1_t* batch,
                             OPJ_UINT32* pool_bytes, OPJ_UINT32* fallbacks);

/* Process-wide number of code-blocks encoded on the GPU so far */
OPJ_UINT64 opj_metal_t1_gpu_cblk_count(void);

/* Bytes of Metal buffers held by the batches currently in the pool (call   */
/* when no batch is in flight to get the total), and their number, i.e.    */
/* the peak number of batches that were in flight at the same time.        */
OPJ_SIZE_T opj_metal_t1_pool_bytes(OPJ_UINT32* nbatches);

#ifdef __cplusplus
}
#endif

#endif /* OPJ_METAL_T1_H */
