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

/* Metal T1 encoder host side. Compiled with -fobjc-arc. See opj_metal_t1.h
 * for the threading model and the memory per batch. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include "opj_metal_t1.h"
#include "opj_metal_common.h"
#include "opj_metal_t1_src.h" /* generated: opj_metal_t1_src[], _len */

/* Layouts shared with t1_enc.metal */
typedef struct {
    uint32_t data_off;
    uint32_t out_off;
    uint32_t scratch_off;
    uint32_t w_h;
    uint32_t orient_sty;
    uint32_t pass_off;
    uint32_t region;
    uint32_t pad;
} opj_mtl_desc;

typedef struct {
    uint32_t numbps;
    uint32_t totalpasses;
    uint32_t nbytes;
    uint32_t status;
    uint32_t pass_off;
    uint32_t region_end;
    uint32_t pad0;
    uint32_t pad1;
} opj_mtl_res;

enum { ST_OK = 0, ST_PASSES = 1, ST_RETRY = 2, ST_BIG = 3, ST_OUTFULL = 4 };

_Static_assert(sizeof(opj_metal_t1_pass) == 12, "pass layout");
_Static_assert(sizeof(opj_mtl_desc) == 32, "desc layout");
_Static_assert(sizeof(opj_mtl_res) == 32, "res layout");

#define OPJ_MTL_CM_CBLKS 4   /* CM_SG_PER_TG of t1_enc.metal */
#define OPJ_MTL_MQ_TG 64     /* MQ_TG of t1_enc.metal */
#define OPJ_MTL_MIN_REGION 256u /* bytes of a design (b) region, at least */
/* Adaptive reservations per design (b) code-block, process-wide: 1.25 x */
/* the average demand of the last batch, within these bounds.            */
#define OPJ_MTL_POOL_MIN 256u     /* symbol pool bytes (beyond the region) */
#define OPJ_MTL_POOL_INIT 3072u
#define OPJ_MTL_POOL_MAX 32768u
#define OPJ_MTL_PASS_MIN 8u       /* pass records */
#define OPJ_MTL_PASS_INIT 24u
#define OPJ_MTL_DEFAULT_MAX_BATCHES 3
#define OPJ_MTL_MAX_RETRIES 4

/* ------------------------------------------------------------------------ */
/* Process-wide shared state                                                 */
/* ------------------------------------------------------------------------ */

@interface OPJMetalT1Shared : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> psoSerial;
@property(nonatomic, strong) id<MTLComputePipelineState> psoCM;
@property(nonatomic, strong) id<MTLComputePipelineState> psoMQ;
@property(nonatomic, strong) NSMutableArray* freeBatches; /* NSValue */
@property(nonatomic, strong) dispatch_semaphore_t slots;
@end

@implementation OPJMetalT1Shared
@end

static OPJMetalT1Shared* g_shared = nil;
static int g_poison = 0;  /* OPJ_METAL_T1_POISON: fill outputs before runs */
static int g_profile = 0; /* OPJ_METAL_T1_PROFILE: per kernel GPU times */
static int g_debug_fail = 0; /* OPJ_METAL_T1_DEBUG_FAIL: runs fail (tests) */
/* Test hooks: OPJ_METAL_T1_DEBUG_POOL=bytes fixes the symbol pool reserve */
/* per code-block (forces GPU retries), OPJ_METAL_T1_DEBUG_REFUSE_EVERY=N */
/* hands every N-th code-block back to the CPU.                           */
static uint32_t g_debug_pool = 0;
static uint32_t g_debug_refuse = 0;
static _Atomic uint32_t g_pool_per_cblk = OPJ_MTL_POOL_INIT;
static _Atomic uint32_t g_pass_per_cblk = OPJ_MTL_PASS_INIT;
static _Atomic uint64_t g_gpu_cblks = 0;

@interface OPJMetalT1Batch : NSObject
@property(nonatomic, strong) id<MTLBuffer> descBuf;
@property(nonatomic, strong) id<MTLBuffer> dataBuf;
@property(nonatomic, strong) id<MTLBuffer> outBuf;
@property(nonatomic, strong) id<MTLBuffer> scratchBuf;
@property(nonatomic, strong) id<MTLBuffer> resBuf;
@property(nonatomic, strong) id<MTLBuffer> passBuf;
@property(nonatomic, strong) id<MTLBuffer> idxBuf;
@property(nonatomic, strong) id<MTLBuffer> poolBuf;
@property(nonatomic, strong) id<MTLBuffer> countersBuf;
@property(nonatomic, strong) id<MTLBuffer> retryBuf;
@end

@implementation OPJMetalT1Batch
@end

struct opj_metal_t1 {
    void* obj; /* OPJMetalT1Batch, retained through CFBridgingRetain */
    opj_metal_t1_kernel_t kernel;
    /* prepared layout */
    OPJ_UINT32 count;
    OPJ_UINT32 nbig;     /* code-blocks for design (a) */
    OPJ_UINT32 pass_big; /* pass records reserved for them */
    opj_metal_t1_cblk_in* in; /* copy of the geometry (data unused) */
    OPJ_UINT32 in_cap;
    double t_prep0, prep_s, gpu_s, total_s;
    uint32_t pool_used, fallbacks;
};

static void opj_mtl_init_shared(void* unused)
{
    (void)unused;
    @autoreleasepool {
        OPJMetalT1Shared* s = [OPJMetalT1Shared new];
        const char* env;
        long nmax = OPJ_MTL_DEFAULT_MAX_BATCHES;

        g_poison = getenv("OPJ_METAL_T1_POISON") != NULL;
        g_profile = getenv("OPJ_METAL_T1_PROFILE") != NULL;
        g_debug_fail = getenv("OPJ_METAL_T1_DEBUG_FAIL") != NULL;
        env = getenv("OPJ_METAL_T1_DEBUG_POOL");
        g_debug_pool = env ? (uint32_t)atol(env) : 0;
        env = getenv("OPJ_METAL_T1_DEBUG_REFUSE_EVERY");
        g_debug_refuse = env ? (uint32_t)atol(env) : 0;
        env = getenv("OPJ_METAL_T1_MAX_BATCHES");
        if (env && atol(env) > 0) {
            nmax = atol(env);
        }
        s.device = MTLCreateSystemDefaultDevice();
        if (!s.device) {
            fprintf(stderr, "opj_metal_t1: no Metal device\n");
            return;
        }
        s.queue = [s.device newCommandQueueWithMaxCommandBufferCount:
                   (NSUInteger)(2 * nmax + 2)];
        NSString* src = [[NSString alloc] initWithBytes:opj_metal_t1_src
                                                 length:opj_metal_t1_src_len
                                               encoding:NSUTF8StringEncoding];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        /* opts.mathMode = MTLMathModeSafe; */
        NSError* err = nil;
        id<MTLLibrary> lib = [s.device newLibraryWithSource:src options:opts
                                                      error:&err];
        if (!lib) {
            fprintf(stderr, "opj_metal_t1: shader compilation failed: %s\n",
                    err.localizedDescription.UTF8String);
            return;
        }
        s.psoSerial = opj_mtl_pso(s.device, lib, @"t1_enc_cblk_serial");
        s.psoCM = opj_mtl_pso(s.device, lib, @"t1_cm_simd");
        s.psoMQ = opj_mtl_pso(s.device, lib, @"t1_mq_serial");
        if (!s.queue || !s.psoSerial || !s.psoCM || !s.psoMQ) {
            return;
        }
        s.freeBatches = [NSMutableArray new];
        s.slots = dispatch_semaphore_create(nmax);
        g_shared = s;
    }
}

static OPJMetalT1Shared* opj_mtl_shared(void)
{
    static dispatch_once_t once;
    dispatch_once_f(&once, NULL, opj_mtl_init_shared);
    return g_shared;
}

OPJ_BOOL opj_metal_t1_available(void)
{
    return opj_mtl_shared() != nil ? OPJ_TRUE : OPJ_FALSE;
}

OPJ_UINT64 opj_metal_t1_gpu_cblk_count(void)
{
    return atomic_load(&g_gpu_cblks);
}

opj_metal_t1_t* opj_metal_t1_acquire(void)
{
    OPJMetalT1Shared* s = opj_mtl_shared();
    opj_metal_t1_t* b = NULL;
    if (!s) {
        return NULL;
    }
    dispatch_semaphore_wait(s.slots, DISPATCH_TIME_FOREVER);
    @synchronized(s) {
        if (s.freeBatches.count) {
            b = (opj_metal_t1_t*)[s.freeBatches.lastObject pointerValue];
            [s.freeBatches removeLastObject];
        }
    }
    if (!b) {
        @autoreleasepool {
            OPJMetalT1Batch* o = [OPJMetalT1Batch new];
            o.countersBuf = [s.device newBufferWithLength:16
                                                  options:MTLResourceStorageModeShared];
            b = (opj_metal_t1_t*)calloc(1, sizeof(*b));
            if (!b || !o.countersBuf) {
                free(b);
                dispatch_semaphore_signal(s.slots);
                return NULL;
            }
            b->obj = (void*)CFBridgingRetain(o);
        }
    }
    b->kernel = OPJ_METAL_T1_KERNEL_PARALLEL;
    b->count = 0;
    return b;
}

void opj_metal_t1_release(opj_metal_t1_t* b)
{
    OPJMetalT1Shared* s = g_shared;
    if (!b || !s) {
        return;
    }
    @synchronized(s) {
        [s.freeBatches addObject:[NSValue valueWithPointer:b]];
    }
    dispatch_semaphore_signal(s.slots);
}

OPJ_SIZE_T opj_metal_t1_pool_bytes(OPJ_UINT32* nbatches)
{
    OPJMetalT1Shared* s = g_shared;
    OPJ_SIZE_T total = 0;
    OPJ_UINT32 n = 0;
    if (s) {
        @synchronized(s) {
            for (NSValue* v in s.freeBatches) {
                opj_metal_t1_t* b = (opj_metal_t1_t*)v.pointerValue;
                OPJMetalT1Batch* o = (__bridge OPJMetalT1Batch*)b->obj;
                total += o.descBuf.length + o.dataBuf.length + o.outBuf.length +
                         o.scratchBuf.length + o.resBuf.length +
                         o.passBuf.length + o.idxBuf.length +
                         o.poolBuf.length + o.countersBuf.length +
                         o.retryBuf.length;
                n++;
            }
        }
    }
    if (nbatches) {
        *nbatches = n;
    }
    return total;
}

opj_metal_t1_t* opj_metal_t1_create(void)
{
    return opj_metal_t1_acquire();
}

void opj_metal_t1_destroy(opj_metal_t1_t* b)
{
    opj_metal_t1_release(b);
}

OPJ_BOOL opj_metal_t1_set_kernel(opj_metal_t1_t* b,
                                 opj_metal_t1_kernel_t kernel)
{
    b->kernel = kernel;
    return OPJ_TRUE;
}

void opj_metal_t1_last_timing(const opj_metal_t1_t* b, double* prep_s,
                              double* gpu_s, double* total_s)
{
    if (prep_s) {
        *prep_s = b->prep_s;
    }
    if (gpu_s) {
        *gpu_s = b->gpu_s;
    }
    if (total_s) {
        *total_s = b->total_s;
    }
}

void opj_metal_t1_last_stats(const opj_metal_t1_t* b,
                             OPJ_UINT32* pool_bytes, OPJ_UINT32* fallbacks)
{
    if (pool_bytes) {
        *pool_bytes = b->pool_used;
    }
    if (fallbacks) {
        *fallbacks = b->fallbacks;
    }
}

static uint32_t opj_mtl_out_size(uint32_t w, uint32_t h)
{
    /* Same bound as opj_tcd_code_block_enc_allocate_data() plus the fake */
    /* leading byte, rounded to 16 bytes */
    return (74u + w * h * 4u + 1u + 15u) & ~15u;
}

static uint32_t opj_mtl_flags_size(uint32_t w, uint32_t h)
{
    return (w + 2) * ((h + 3) / 4 + 2);
}

static uint32_t opj_mtl_clamp(uint64_t v, uint32_t lo, uint32_t hi)
{
    return v < lo ? lo : v > hi ? hi : (uint32_t)v;
}

OPJ_BOOL opj_metal_t1_prepare(opj_metal_t1_t* b,
                              const opj_metal_t1_cblk_in* in,
                              OPJ_UINT32 count, OPJ_INT32** data_ptrs)
{
    OPJMetalT1Shared* s = g_shared;
    OPJMetalT1Batch* o = (__bridge OPJMetalT1Batch*)b->obj;
    const BOOL serial = b->kernel == OPJ_METAL_T1_KERNEL_SERIAL;
    size_t ndata = 0, nout = 0, nscratch = 0, npass;
    OPJ_UINT32 i, nbig = 0;
    id<MTLDevice> dev = s.device;
    id<MTLBuffer> buf;

    b->t_prep0 = opj_mtl_now();
    b->count = 0;
    for (i = 0; i < count; ++i) {
        const opj_metal_t1_cblk_in* c = &in[i];
        uint32_t bytes;
        if (c->cblksty != 0 || c->w == 0 || c->h == 0 ||
                c->w > OPJ_METAL_T1_MAX_SIDE || c->h > OPJ_METAL_T1_MAX_SIDE ||
                c->w * c->h > OPJ_METAL_T1_MAX_AREA) {
            return OPJ_FALSE;
        }
        bytes = (c->w * c->h * 4u + 15u) & ~15u;
        if (serial || c->w > 32 || c->h > 32) {
            nout += opj_mtl_out_size(c->w, c->h);
            nscratch += opj_mtl_flags_size(c->w, c->h);
            nbig++;
        } else if (bytes < OPJ_MTL_MIN_REGION) {
            bytes = OPJ_MTL_MIN_REGION;
        }
        ndata += bytes;
    }
    if (ndata > 0xfffffff0u || nout > 0xffffffffu || nscratch > 0x3fffffffu) {
        return OPJ_FALSE;
    }
    npass = (size_t)nbig * OPJ_METAL_T1_MAXPASSES +
            (size_t)(count - nbig) * atomic_load(&g_pass_per_cblk);
    if (b->in_cap < count) {
        opj_metal_t1_cblk_in* p = (opj_metal_t1_cblk_in*)realloc(b->in,
                                  (size_t)count * sizeof(*p));
        if (!p) {
            return OPJ_FALSE;
        }
        b->in = p;
        b->in_cap = count;
    }
    @autoreleasepool {
#define OPJ_MTL_RESERVE(field, size) \
        buf = o.field; \
        if (!opj_mtl_reserve(dev, &buf, (size))) { \
            return OPJ_FALSE; \
        } \
        o.field = buf;
        OPJ_MTL_RESERVE(descBuf, (size_t)count * sizeof(opj_mtl_desc));
        OPJ_MTL_RESERVE(dataBuf, ndata);
        OPJ_MTL_RESERVE(outBuf, nout);
        OPJ_MTL_RESERVE(scratchBuf, nscratch * 4);
        OPJ_MTL_RESERVE(resBuf, (size_t)count * sizeof(opj_mtl_res));
        OPJ_MTL_RESERVE(passBuf, npass * sizeof(opj_metal_t1_pass));
        OPJ_MTL_RESERVE(idxBuf, (size_t)(nbig ? nbig : 1) * 4);
        OPJ_MTL_RESERVE(poolBuf, (size_t)(count - nbig) *
                        (g_debug_pool ? g_debug_pool :
                         atomic_load(&g_pool_per_cblk)));
#undef OPJ_MTL_RESERVE
    }

    opj_mtl_desc* desc = (opj_mtl_desc*)o.descBuf.contents;
    uint8_t* data = (uint8_t*)o.dataBuf.contents;
    uint32_t* idx = (uint32_t*)o.idxBuf.contents;
    uint32_t doff = 0, ooff = 0, soff = 0, nidx = 0;
    for (i = 0; i < count; ++i) {
        const opj_metal_t1_cblk_in* c = &in[i];
        uint32_t bytes = (c->w * c->h * 4u + 15u) & ~15u;
        memset(&desc[i], 0, sizeof(desc[i]));
        desc[i].data_off = doff / 4;
        desc[i].w_h = c->w | (c->h << 16);
        desc[i].orient_sty = c->orient | (c->cblksty << 8);
        if (serial || c->w > 32 || c->h > 32) {
            desc[i].out_off = ooff;
            desc[i].scratch_off = soff;
            desc[i].pass_off = nidx * OPJ_METAL_T1_MAXPASSES;
            ooff += opj_mtl_out_size(c->w, c->h);
            soff += opj_mtl_flags_size(c->w, c->h);
            idx[nidx++] = i;
        } else if (bytes < OPJ_MTL_MIN_REGION) {
            bytes = OPJ_MTL_MIN_REGION;
        }
        desc[i].region = bytes;
        if (data_ptrs) {
            data_ptrs[i] = (OPJ_INT32*)(data + doff);
        }
        b->in[i] = *c;
        b->in[i].data = NULL;
        doff += bytes;
    }
    b->count = count;
    b->nbig = nbig;
    b->pass_big = nbig * OPJ_METAL_T1_MAXPASSES;
    if (g_poison) {
        /* Debug: make stale results impossible to mistake for new ones */
        memset(o.outBuf.contents, 0xa5, o.outBuf.length);
        memset(o.resBuf.contents, 0xa5, o.resBuf.length);
        memset(o.passBuf.contents, 0xa5, o.passBuf.length);
        memset(o.poolBuf.contents, 0xa5, o.poolBuf.length);
    }
    b->prep_s = opj_mtl_now() - b->t_prep0;
    return OPJ_TRUE;
}

static void opj_mtl_encode_serial(id<MTLComputeCommandEncoder> enc,
                                  OPJMetalT1Shared* s, OPJMetalT1Batch* o,
                                  uint32_t n, uint32_t use_idx)
{
    [enc setComputePipelineState:s.psoSerial];
    [enc setBuffer:o.descBuf offset:0 atIndex:0];
    [enc setBuffer:o.dataBuf offset:0 atIndex:1];
    [enc setBuffer:o.outBuf offset:0 atIndex:2];
    [enc setBuffer:o.scratchBuf offset:0 atIndex:3];
    [enc setBuffer:o.resBuf offset:0 atIndex:4];
    [enc setBuffer:o.passBuf offset:0 atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];
    [enc setBuffer:o.idxBuf offset:0 atIndex:7];
    [enc setBytes:&use_idx length:sizeof(use_idx) atIndex:8];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

/* Design (b) context modelling + MQ over n code-blocks (idx list or all) */
static void opj_mtl_encode_parallel(id<MTLComputeCommandEncoder> enc,
                                    OPJMetalT1Shared* s, OPJMetalT1Batch* o,
                                    uint32_t n, id<MTLBuffer> idxBuf,
                                    uint32_t use_idx, BOOL split_profile,
                                    id<MTLCommandBuffer> __strong* cbp,
                                    id<MTLComputeCommandEncoder> __strong* encp,
                                    double* gpu_s)
{
    uint32_t caps[2];
    caps[0] = (uint32_t)(o.poolBuf.length > 0xffffff00u ? 0xffffff00u :
                         o.poolBuf.length);
    caps[1] = (uint32_t)(o.passBuf.length / sizeof(opj_metal_t1_pass));
    [enc setComputePipelineState:s.psoCM];
    [enc setBuffer:o.descBuf offset:0 atIndex:0];
    [enc setBuffer:o.dataBuf offset:0 atIndex:1];
    [enc setBuffer:o.poolBuf offset:0 atIndex:2];
    [enc setBuffer:o.countersBuf offset:0 atIndex:3];
    [enc setBuffer:o.resBuf offset:0 atIndex:4];
    [enc setBuffer:o.passBuf offset:0 atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];
    [enc setBytes:caps length:sizeof(caps) atIndex:7];
    [enc setBuffer:idxBuf offset:0 atIndex:8];
    [enc setBytes:&use_idx length:sizeof(use_idx) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake((n + OPJ_MTL_CM_CBLKS - 1) /
                                          OPJ_MTL_CM_CBLKS, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32 * OPJ_MTL_CM_CBLKS, 1, 1)];
    if (split_profile) {
        /* Debug: time the context modelling kernel on its own */
        id<MTLCommandBuffer> cb = *cbp;
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        fprintf(stderr, "opj_metal_t1: t1_cm_simd %.2f ms\n",
                1e3 * (cb.GPUEndTime - cb.GPUStartTime));
        *gpu_s += cb.GPUEndTime - cb.GPUStartTime;
        *cbp = [s.queue commandBufferWithUnretainedReferences];
        *encp = [*cbp computeCommandEncoder];
        enc = *encp;
        [enc setBuffer:o.descBuf offset:0 atIndex:2];
        [enc setBuffer:o.resBuf offset:0 atIndex:4];
        [enc setBuffer:o.passBuf offset:0 atIndex:5];
        [enc setBytes:&n length:sizeof(n) atIndex:6];
        [enc setBuffer:idxBuf offset:0 atIndex:8];
        [enc setBytes:&use_idx length:sizeof(use_idx) atIndex:9];
    }
    [enc setComputePipelineState:s.psoMQ];
    [enc setBuffer:o.dataBuf offset:0 atIndex:0];
    [enc setBuffer:o.poolBuf offset:0 atIndex:1];
    [enc setBuffer:o.descBuf offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(OPJ_MTL_MQ_TG, 1, 1)];
}

OPJ_BOOL opj_metal_t1_run(opj_metal_t1_t* b, opj_metal_t1_cblk_out* out)
{
    OPJMetalT1Shared* s = g_shared;
    const BOOL serial = b->kernel == OPJ_METAL_T1_KERNEL_SERIAL;
    const uint32_t count = b->count;
    const uint32_t nsmall = count - b->nbig;
    uint32_t i;
    double gpu_s = 0;

    b->gpu_s = 0;
    b->pool_used = 0;
    b->fallbacks = 0;
    if (count == 0) {
        b->total_s = opj_mtl_now() - b->t_prep0;
        return OPJ_TRUE;
    }
    @autoreleasepool {
        OPJMetalT1Batch* o = (__bridge OPJMetalT1Batch*)b->obj;
        opj_mtl_desc* desc = (opj_mtl_desc*)o.descBuf.contents;
        opj_mtl_res* r = (opj_mtl_res*)o.resBuf.contents;
        uint32_t* counters = (uint32_t*)o.countersBuf.contents;
        uint64_t pool_demand = 0;
        int round;

        counters[0] = 0;
        counters[1] = b->pass_big;
        id<MTLCommandBuffer> cb = [s.queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            return OPJ_FALSE;
        }
        if (!serial && nsmall) {
            opj_mtl_encode_parallel(enc, s, o, count, o.descBuf, 0, g_profile,
                                    &cb, &enc, &gpu_s);
        }
        if (b->nbig) {
            opj_mtl_encode_serial(enc, s, o, b->nbig, serial ? 0 : 1);
        }
        [enc endEncoding];
        [cb commit];
        if (!opj_mtl_wait(cb)) {
            return OPJ_FALSE;
        }
        gpu_s += cb.GPUEndTime - cb.GPUStartTime;
        if (g_profile) {
            fprintf(stderr, "opj_metal_t1: %s %.2f ms\n",
                    serial ? "t1_enc_cblk_serial" : "t1_mq_serial (+serial part)",
                    1e3 * (cb.GPUEndTime - cb.GPUStartTime));
        }

        /* Pool or pass records full: those code-blocks got their input */
        /* back; run them again with bigger buffers. */
        for (round = 0; !serial && nsmall && round <= OPJ_MTL_MAX_RETRIES; ++round) {
            uint32_t nretry = 0, *ridx;
            size_t need_pass;
            id<MTLBuffer> buf;
            pool_demand += counters[0];
            for (i = 0; i < count; ++i) {
                if (r[i].status == ST_RETRY) {
                    nretry++;
                }
            }
            if (nretry == 0 || round == OPJ_MTL_MAX_RETRIES) {
                break;
            }
            b->fallbacks += nretry;
            buf = o.retryBuf;
            if (!opj_mtl_reserve(s.device, &buf, (size_t)nretry * 4)) {
                return OPJ_FALSE;
            }
            o.retryBuf = buf;
            ridx = (uint32_t*)o.retryBuf.contents;
            nretry = 0;
            for (i = 0; i < count; ++i) {
                if (r[i].status == ST_RETRY) {
                    ridx[nretry++] = i;
                }
            }
            /* the symbol pool is free again; the pass records are kept */
            {
                size_t pool = o.poolBuf.length * 2;
                size_t want = (size_t)nretry * OPJ_MTL_POOL_MAX;
                if (pool < (size_t)nretry * 8192u) {
                    pool = (size_t)nretry * 8192u;
                }
                buf = o.poolBuf;
                if (pool > want) {
                    pool = want;
                }
                if (pool < o.poolBuf.length) {
                    pool = o.poolBuf.length;
                }
                if (pool > o.poolBuf.length) {
                    buf = nil;
                    o.poolBuf = nil;
                    if (!opj_mtl_reserve(s.device, &buf, pool)) {
                        return OPJ_FALSE;
                    }
                    o.poolBuf = buf;
                }
            }
            need_pass = (size_t)counters[1] + (size_t)nretry * OPJ_METAL_T1_MAXPASSES;
            if (need_pass * sizeof(opj_metal_t1_pass) > o.passBuf.length) {
                id<MTLBuffer> nb = [s.device newBufferWithLength:need_pass *
                                    sizeof(opj_metal_t1_pass)
                                    options:MTLResourceStorageModeShared];
                if (!nb) {
                    return OPJ_FALSE;
                }
                memcpy(nb.contents, o.passBuf.contents, o.passBuf.length);
                o.passBuf = nb;
            }
            {
                uint32_t cap = (uint32_t)(o.passBuf.length / sizeof(opj_metal_t1_pass));
                if (counters[1] > cap) {
                    counters[1] = cap;
                }
            }
            counters[0] = 0;
            cb = [s.queue commandBufferWithUnretainedReferences];
            enc = [cb computeCommandEncoder];
            opj_mtl_encode_parallel(enc, s, o, nretry, o.retryBuf, 1, NO, &cb, &enc,
                                    &gpu_s);
            [enc endEncoding];
            [cb commit];
            if (!opj_mtl_wait(cb)) {
                return OPJ_FALSE;
            }
            gpu_s += cb.GPUEndTime - cb.GPUStartTime;
        }
        if (!serial && nsmall) {
            /* adapt the reservations of the next batches */
            atomic_store(&g_pool_per_cblk,
                         opj_mtl_clamp(pool_demand * 5 / 4 / nsmall,
                                       OPJ_MTL_POOL_MIN, OPJ_MTL_POOL_MAX));
            atomic_store(&g_pass_per_cblk,
                         opj_mtl_clamp(((uint64_t)counters[1] - b->pass_big) * 5 / 4 /
                                       nsmall + 1, OPJ_MTL_PASS_MIN,
                                       OPJ_METAL_T1_MAXPASSES));
        }
        b->pool_used = (uint32_t)(pool_demand > 0xffffffffu ? 0xffffffffu :
                                  pool_demand);

        const uint8_t* ob = (const uint8_t*)o.outBuf.contents;
        const uint8_t* db = (const uint8_t*)o.dataBuf.contents;
        const opj_metal_t1_pass* pb = (const opj_metal_t1_pass*)o.passBuf.contents;
        uint32_t ngpu = 0;
        for (i = 0; i < count; ++i) {
            const BOOL big = serial || b->in[i].w > 32 || b->in[i].h > 32;
            if (big) {
                out[i].data = ob + desc[i].out_off + 1;
                out[i].data_max = opj_mtl_out_size(b->in[i].w, b->in[i].h) - 1;
            } else {
                out[i].data = db + (size_t)desc[i].data_off * 4 + 1;
                out[i].data_max = desc[i].region - 1;
            }
            out[i].nbytes = r[i].nbytes;
            out[i].numbps = r[i].numbps;
            out[i].totalpasses = r[i].totalpasses;
            out[i].passes = pb + r[i].pass_off;
            out[i].status = 0;
            if (r[i].status != ST_OK ||
                    r[i].totalpasses > OPJ_METAL_T1_MAXPASSES ||
                    r[i].nbytes > out[i].data_max ||
                    (g_debug_refuse && i % g_debug_refuse == 0)) {
                out[i].status = 1;
                out[i].totalpasses = 0;
                out[i].passes = pb;
            } else {
                ngpu++;
            }
        }
        if (g_debug_fail) {
            /* Test hook: report a failure after the GPU wrote its outputs */
            return OPJ_FALSE;
        }
        atomic_fetch_add(&g_gpu_cblks, ngpu);
        b->gpu_s = gpu_s;
        b->total_s = opj_mtl_now() - b->t_prep0;
        return OPJ_TRUE;
    }
}

OPJ_BOOL opj_metal_t1_encode_batch(opj_metal_t1_t* b,
                                   const opj_metal_t1_cblk_in* in,
                                   opj_metal_t1_cblk_out* out,
                                   OPJ_UINT32 count)
{
    OPJ_INT32** ptrs;
    OPJ_UINT32 i;
    double t0 = opj_mtl_now(), t_copy;
    ptrs = (OPJ_INT32**)malloc((count ? count : 1) * sizeof(*ptrs));
    if (!ptrs) {
        return OPJ_FALSE;
    }
    if (!opj_metal_t1_prepare(b, in, count, ptrs)) {
        free(ptrs);
        return OPJ_FALSE;
    }
    for (i = 0; i < count; ++i) {
        memcpy(ptrs[i], in[i].data, (size_t)in[i].w * in[i].h * 4);
    }
    free(ptrs);
    t_copy = opj_mtl_now();
    b->t_prep0 = t0;
    b->prep_s = t_copy - t0;
    return opj_metal_t1_run(b, out);
}
