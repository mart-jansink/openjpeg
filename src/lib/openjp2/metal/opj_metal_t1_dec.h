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
 * JPEG 2000 Tier-1 (EBCOT) code-block decoder on Apple GPUs through Metal.
 * The decoded samples are bit-identical to those of opj_t1_decode_cblk()
 * followed by the dequantisation of opj_t1_clbl_decode_processor(), for
 * code-block style 0 (no BYPASS, RESET, TERMALL, VSC, PTERM, SEGSYM, HT),
 * no ROI shift and a single codeword segment.
 *
 * Plain C API; the implementation is Objective-C (opj_metal_t1_dec.m).
 */

#ifndef OPJ_METAL_T1_DEC_H
#define OPJ_METAL_T1_DEC_H

#include "openjpeg.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Largest code-block side and area the GPU decoder takes */
#define OPJ_METAL_T1_DEC_MAX_SIDE 64
#define OPJ_METAL_T1_DEC_MAX_AREA 4096
/* Largest numbps (bit-planes) the GPU decoder takes: as opj_t1_decode_cblk() */
#define OPJ_METAL_T1_DEC_MAX_NUMBPS 30
/* Planes that can be mapped at once (opj_metal_t1_dec_set_planes()) */
#define OPJ_METAL_T1_DEC_MAX_PLANES 4

typedef struct opj_metal_t1_dec opj_metal_t1_dec_t;

typedef struct opj_metal_t1_dec_cblk_in {
    OPJ_UINT32 w;
    OPJ_UINT32 h;
    OPJ_UINT32 orient;      /* band->bandno: 0 LL, 1 HL, 2 LH, 3 HH */
    OPJ_UINT32 qmfbid;      /* 1: 5/3 (output v / 2), 0: 9/7 (float output) */
    OPJ_UINT32 numbps;      /* cblk->numbps (bpno_plus_one, roishift 0) */
    OPJ_UINT32 npasses;     /* passes to decode, 0 for an empty code-block */
    OPJ_UINT32 len;         /* coded bytes of the (single) segment */
    OPJ_FLOAT32 half_step;  /* 9/7: 0.5f * band->stepsize */
    /* Destination: -1 for a w*h raster region owned by the batch         */
    /* (opj_metal_t1_dec_prepare() returns it), otherwise the index of a  */
    /* plane mapped with opj_metal_t1_dec_set_planes(), out_off the index */
    /* of the top-left sample in it and out_stride its samples per row.   */
    OPJ_INT32 plane;
    OPJ_UINT32 out_off;
    OPJ_UINT32 out_stride;
} opj_metal_t1_dec_cblk_in;

/*
 * Threading model and memory: as the encoder (opj_metal_t1.h). The device,
 * library and pipeline are process-wide, created lazily and thread-safely
 * on first use; work goes through batches taken from a process-wide pool.
 * Admission never blocks: try_acquire fails while the caller's limit of
 * command buffers is in flight, so a decoder that finds the GPU busy
 * decodes on the CPU instead of waiting (waiting leaves CPU cores idle). A
 * slot frees when its command buffer completes. Each batch has its own
 * command queue, so batches of parallel decoders overlap on the GPU.
 * OPJ_METAL_T1_DEC_INFLIGHT overrides the callers' limits. The pool holds
 * OPJ_METAL_T1_DEC_MAX_BATCHES batches (default 5). Released batches keep
 * their buffers. Per code-block a batch holds its coded bytes (+ 2..5 bytes), a
 * 48-byte descriptor, a 4-byte status and its flags ((w+2) * ((h+3)/4+2) *
 * 4 bytes), plus a w*h*4 byte output region unless it is decoded straight
 * into a mapped plane.
 *
 * Environment (debugging and tests): OPJ_METAL_T1_DEC_PROFILE prints GPU times,
 * OPJ_METAL_T1_DEC_POISON fills the outputs before each run,
 * OPJ_METAL_T1_DEC_DEBUG_FAIL makes runs fail after the GPU wrote its
 * outputs, OPJ_METAL_T1_DEC_DEBUG_REFUSE_EVERY=N hands every N-th
 * code-block back to the caller.
 */

/* OPJ_TRUE if a Metal device is available and the kernels compiled */
OPJ_BOOL opj_metal_t1_dec_available(void);

/* Takes a batch from the pool and a GPU slot, without blocking; NULL when  */
/* max_inflight (1 to 3) command buffers are already in flight, or Metal is */
/* not available */
opj_metal_t1_dec_t* opj_metal_t1_dec_try_acquire(OPJ_UINT32 max_inflight);

/* Returns a batch to the pool; its outputs and mapped planes go away */
void opj_metal_t1_dec_release(opj_metal_t1_dec_t* batch);

/* Maps n caller-owned planes (page-aligned, bytes[i] a multiple of the     */
/* page size) for the next run, without copying: the code-blocks that name  */
/* a plane are decoded straight into it (all samples of their regions are   */
/* written). Returns OPJ_FALSE when one cannot be mapped; no plane is       */
/* mapped then. n == 0 unmaps all.                                          */
OPJ_BOOL opj_metal_t1_dec_set_planes(opj_metal_t1_dec_t* batch,
                                     OPJ_INT32* const* planes,
                                     const OPJ_SIZE_T* bytes, OPJ_UINT32 n);

/* Lays out count code-blocks. On success in_ptrs[i] is where the caller   */
/* must write the in[i].len coded bytes (the end-of-data marker behind them */
/* is already there) and out_ptrs[i], for in[i].plane == -1, where the w*h */
/* decoded samples (raster order, int32 or float32 bit patterns) are after */
/* opj_metal_t1_dec_wait(), NULL otherwise. Returns OPJ_FALSE on an         */
/* unsupported code-block or an allocation failure.                        */
OPJ_BOOL opj_metal_t1_dec_prepare(opj_metal_t1_dec_t* batch,
                                  const opj_metal_t1_dec_cblk_in* in,
                                  OPJ_UINT32 count, OPJ_BYTE** in_ptrs,
                                  OPJ_INT32** out_ptrs);

/* Starts decoding the prepared code-blocks (one command buffer). Until    */
/* opj_metal_t1_dec_wait() the caller may only write samples of the mapped */
/* planes outside the regions of the code-blocks of the batch. OPJ_FALSE  */
/* on a Metal failure: nothing was started, call wait() all the same.     */
OPJ_BOOL opj_metal_t1_dec_commit(opj_metal_t1_dec_t* batch);

/* Waits for the decoding started by opj_metal_t1_dec_commit(). status[i] */
/* (count entries) is 0 when code-block i was decoded, otherwise the      */
/* caller must decode it on the CPU. OPJ_FALSE on a Metal failure: no     */
/* output is valid then (mapped planes may have been partly written).     */
OPJ_BOOL opj_metal_t1_dec_wait(opj_metal_t1_dec_t* batch, OPJ_UINT32* status);

/* Timing of the last prepare, commit and wait, in seconds: host            */
/* preparation, GPU execution (GPUStartTime..GPUEndTime) and the wall time */
/* from commit to the end of wait.                                          */
void opj_metal_t1_dec_last_timing(const opj_metal_t1_dec_t* batch,
                                  double* prep_s, double* gpu_s,
                                  double* run_s);

/* Process-wide number of code-blocks decoded on the GPU so far */
OPJ_UINT64 opj_metal_t1_dec_gpu_cblk_count(void);

/* Bytes of Metal buffers held by the batches in the pool, and their number */
OPJ_SIZE_T opj_metal_t1_dec_pool_bytes(OPJ_UINT32* nbatches);

#ifdef __cplusplus
}
#endif

#endif /* OPJ_METAL_T1_DEC_H */
