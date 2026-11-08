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


/* Metal T1 decoder host side. Compiled with -fobjc-arc. See
 * opj_metal_t1_dec.h for the threading model and the memory per batch. */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

#include "opj_metal_t1_dec.h"
#include "opj_metal_common.h"
#include "opj_metal_t1_dec_src.h" /* generated: opj_metal_t1_dec_src[], _len */

/* Layout shared with t1_dec.metal */
typedef struct {
    uint32_t in_off;
    uint32_t len;
    uint32_t w_h;
    uint32_t orient_qmf;
    uint32_t bps_passes;
    uint32_t out_off;
    uint32_t out_stride;
    uint32_t half_step;
    uint32_t scratch_off;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
} opj_mtl_dec_desc;

_Static_assert(sizeof(opj_mtl_dec_desc) == 48, "desc layout");

#define OPJ_MTL_DEC_TG 64    /* threads per threadgroup: DEC_TG of t1_dec.metal */
#define OPJ_MTL_DEC_SIMD 32  /* SIMD-group width of Apple GPUs */
#define OPJ_MTL_DEC_MAX_INFLIGHT 3   /* GPU command buffers in flight, at most */
#define OPJ_MTL_DEC_SPARE_BATCHES 2  /* batches beyond that, see try_acquire */
#define OPJ_MTL_DEC_HOLE 0xffffffffu

/* ------------------------------------------------------------------------ */
/* Process-wide shared state                                                 */
/* ------------------------------------------------------------------------ */

@interface OPJMetalT1DecShared : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue; /* if a batch has none */
@property(nonatomic, strong) id<MTLComputePipelineState> pso;
@property(nonatomic, strong) NSMutableArray* freeBatches; /* NSValue */
@property(nonatomic, strong) dispatch_semaphore_t slots;
@end

@implementation OPJMetalT1DecShared
@end

static OPJMetalT1DecShared* g_shared = nil;
static int g_poison = 0;
static int g_profile = 0;
static int g_debug_fail = 0;
static uint32_t g_debug_refuse = 0;
static _Atomic uint64_t g_gpu_cblks = 0;
/* GPU admission: command buffers committed or about to be, freed by their */
/* completion handlers; try_acquire() fails at the caller's limit */
static _Atomic int g_inflight = 0;
static int g_inflight_env = 0; /* OPJ_METAL_T1_DEC_INFLIGHT, 0: unset */

@interface OPJMetalT1DecBatch : NSObject
@property(nonatomic, strong) id<MTLBuffer> descBuf;
@property(nonatomic, strong) id<MTLBuffer> inBuf;
@property(nonatomic, strong) id<MTLBuffer> outBuf;
@property(nonatomic, strong) id<MTLBuffer> scratchBuf;
@property(nonatomic, strong) id<MTLBuffer> statusBuf;
@property(nonatomic, strong) id<MTLBuffer> orderBuf;
@property(nonatomic, strong) NSMutableArray* planes; /* id<MTLBuffer> */
@property(nonatomic, strong) id<MTLCommandBuffer> cb; /* committed */
/* Its own queue, so that the batches of parallel decoders can run on the */
/* GPU at the same time (one 2K tile leaves the GPU under-occupied: twice  */
/* the code-blocks in one dispatch took 1.44 times as long) */
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@end

@implementation OPJMetalT1DecBatch
@end

struct opj_metal_t1_dec {
    void* obj; /* OPJMetalT1DecBatch, retained through CFBridgingRetain */
    OPJ_UINT32 count;     /* code-blocks */
    OPJ_UINT32 nthreads;  /* dispatch positions, holes included */
    OPJ_UINT32 nplanes;
    size_t scratch_bytes; /* in use, zeroed by commit() */
    double t_commit;
    int slot; /* holds a GPU admission slot not yet handed to a command buffer */
    double prep_s, gpu_s, run_s;
    uint64_t* keys; /* sort scratch */
    OPJ_UINT32 keys_cap;
};

static void opj_mtl_dec_init_shared(void* unused)
{
    (void)unused;
    @autoreleasepool {
        OPJMetalT1DecShared* s = [OPJMetalT1DecShared new];
        const char* env;
        long nmax;

        g_poison = getenv("OPJ_METAL_T1_DEC_POISON") != NULL;
        g_profile = getenv("OPJ_METAL_T1_DEC_PROFILE") != NULL;
        g_debug_fail = getenv("OPJ_METAL_T1_DEC_DEBUG_FAIL") != NULL;
        env = getenv("OPJ_METAL_T1_DEC_DEBUG_REFUSE_EVERY");
        g_debug_refuse = env ? (uint32_t)atol(env) : 0;
        env = getenv("OPJ_METAL_T1_DEC_INFLIGHT");
        if (env && atoi(env) > 0) {
            g_inflight_env = atoi(env);
        }
        nmax = (g_inflight_env > OPJ_MTL_DEC_MAX_INFLIGHT ? g_inflight_env :
                OPJ_MTL_DEC_MAX_INFLIGHT) + OPJ_MTL_DEC_SPARE_BATCHES;
        env = getenv("OPJ_METAL_T1_DEC_MAX_BATCHES");
        if (env && atol(env) > 0) {
            nmax = atol(env);
        }
        s.device = MTLCreateSystemDefaultDevice();
        if (!s.device) {
            fprintf(stderr, "opj_metal_t1_dec: no Metal device\n");
            return;
        }
        s.queue = [s.device newCommandQueueWithMaxCommandBufferCount:
                   (NSUInteger)(nmax + 2)];
        NSString* src = [[NSString alloc] initWithBytes:opj_metal_t1_dec_src
                                                 length:opj_metal_t1_dec_src_len
                                               encoding:NSUTF8StringEncoding];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        NSError* err = nil;
        id<MTLLibrary> lib = [s.device newLibraryWithSource:src options:opts
                                                      error:&err];
        if (!lib) {
            fprintf(stderr, "opj_metal_t1_dec: shader compilation failed: %s\n",
                    err.localizedDescription.UTF8String);
            return;
        }
        s.pso = opj_mtl_pso(s.device, lib, @"t1_dec_cblk");
        if (!s.queue || !s.pso) {
            return;
        }
        if (s.pso.maxTotalThreadsPerThreadgroup < OPJ_MTL_DEC_TG ||
                s.pso.threadExecutionWidth != OPJ_MTL_DEC_SIMD) {
            fprintf(stderr, "opj_metal_t1_dec: unsupported GPU (threadgroup "
                    "%lu, SIMD width %lu)\n",
                    (unsigned long)s.pso.maxTotalThreadsPerThreadgroup,
                    (unsigned long)s.pso.threadExecutionWidth);
            return;
        }
        s.freeBatches = [NSMutableArray new];
        s.slots = dispatch_semaphore_create(nmax);
        g_shared = s;
    }
}

static OPJMetalT1DecShared* opj_mtl_dec_shared(void)
{
    static dispatch_once_t once;
    dispatch_once_f(&once, NULL, opj_mtl_dec_init_shared);
    return g_shared;
}

OPJ_BOOL opj_metal_t1_dec_available(void)
{
    return opj_mtl_dec_shared() != nil ? OPJ_TRUE : OPJ_FALSE;
}

OPJ_UINT64 opj_metal_t1_dec_gpu_cblk_count(void)
{
    return atomic_load(&g_gpu_cblks);
}

opj_metal_t1_dec_t* opj_metal_t1_dec_try_acquire(OPJ_UINT32 max_inflight)
{
    OPJMetalT1DecShared* s = opj_mtl_dec_shared();
    opj_metal_t1_dec_t* b = NULL;
    int cur, limit;
    if (!s) {
        return NULL;
    }
    limit = g_inflight_env > 0 ? g_inflight_env :
            (int)(max_inflight < 1 ? 1 : max_inflight > OPJ_MTL_DEC_MAX_INFLIGHT ?
                  OPJ_MTL_DEC_MAX_INFLIGHT : max_inflight);
    /* A GPU slot first: never wait for the GPU, the CPU decodes instead */
    cur = atomic_load(&g_inflight);
    do {
        if (cur >= limit) {
            return NULL;
        }
    } while (!atomic_compare_exchange_weak(&g_inflight, &cur, cur + 1));
    /* Then a batch (memory); one is normally free, since batches are */
    /* returned shortly after their GPU run */
    if (dispatch_semaphore_wait(s.slots, DISPATCH_TIME_NOW) != 0) {
        atomic_fetch_sub(&g_inflight, 1);
        return NULL;
    }
    @synchronized(s) {
        if (s.freeBatches.count) {
            b = (opj_metal_t1_dec_t*)[s.freeBatches.lastObject pointerValue];
            [s.freeBatches removeLastObject];
        }
    }
    if (!b) {
        @autoreleasepool {
            OPJMetalT1DecBatch* o = [OPJMetalT1DecBatch new];
            o.planes = [NSMutableArray new];
            b = (opj_metal_t1_dec_t*)calloc(1, sizeof(*b));
            if (!b) {
                dispatch_semaphore_signal(s.slots);
                atomic_fetch_sub(&g_inflight, 1);
                return NULL;
            }
            b->obj = (void*)CFBridgingRetain(o);
        }
    }
    b->count = 0;
    b->nthreads = 0;
    b->nplanes = 0;
    b->slot = 1;
    return b;
}

void opj_metal_t1_dec_release(opj_metal_t1_dec_t* b)
{
    OPJMetalT1DecShared* s = g_shared;
    if (!b || !s) {
        return;
    }
    {
        OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
        /* the planes belong to the caller: never keep them mapped */
        @autoreleasepool {
            if (o.cb) {
                [o.cb waitUntilCompleted]; /* committed, never waited for */
                o.cb = nil;
            }
            [o.planes removeAllObjects];
        }
        b->nplanes = 0;
    }
    if (b->slot) {
        /* never committed */
        b->slot = 0;
        atomic_fetch_sub(&g_inflight, 1);
    }
    @synchronized(s) {
        [s.freeBatches addObject:[NSValue valueWithPointer:b]];
    }
    dispatch_semaphore_signal(s.slots);
}

OPJ_SIZE_T opj_metal_t1_dec_pool_bytes(OPJ_UINT32* nbatches)
{
    OPJMetalT1DecShared* s = g_shared;
    OPJ_SIZE_T total = 0;
    OPJ_UINT32 n = 0;
    if (s) {
        @synchronized(s) {
            for (NSValue* v in s.freeBatches) {
                opj_metal_t1_dec_t* b = (opj_metal_t1_dec_t*)v.pointerValue;
                OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
                total += o.descBuf.length + o.inBuf.length + o.outBuf.length +
                         o.scratchBuf.length + o.statusBuf.length +
                         o.orderBuf.length;
                n++;
            }
        }
    }
    if (nbatches) {
        *nbatches = n;
    }
    return total;
}

void opj_metal_t1_dec_last_timing(const opj_metal_t1_dec_t* b, double* prep_s,
                                  double* gpu_s, double* run_s)
{
    if (prep_s) {
        *prep_s = b->prep_s;
    }
    if (gpu_s) {
        *gpu_s = b->gpu_s;
    }
    if (run_s) {
        *run_s = b->run_s;
    }
}

OPJ_BOOL opj_metal_t1_dec_set_planes(opj_metal_t1_dec_t* b,
                                     OPJ_INT32* const* planes,
                                     const OPJ_SIZE_T* bytes, OPJ_UINT32 n)
{
    OPJMetalT1DecShared* s = g_shared;
    OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
    const size_t page = (size_t)getpagesize();
    OPJ_UINT32 i;
    @autoreleasepool {
        [o.planes removeAllObjects];
        b->nplanes = 0;
        if (n > OPJ_METAL_T1_DEC_MAX_PLANES) {
            return OPJ_FALSE;
        }
        for (i = 0; i < n; ++i) {
            id<MTLBuffer> buf;
            if (((size_t)planes[i] & (page - 1)) != 0 || bytes[i] == 0 ||
                    (bytes[i] & (page - 1)) != 0) {
                [o.planes removeAllObjects];
                return OPJ_FALSE;
            }
            buf = [s.device newBufferWithBytesNoCopy:planes[i] length:bytes[i]
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
            if (!buf) {
                [o.planes removeAllObjects];
                return OPJ_FALSE;
            }
            [o.planes addObject:buf];
        }
        b->nplanes = n;
    }
    return OPJ_TRUE;
}

static uint32_t opj_mtl_dec_flags_size(uint32_t w, uint32_t h)
{
    return (w + 2) * ((h + 3) / 4 + 2);
}

static int opj_mtl_dec_cmp_key(const void* a, const void* b)
{
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x < y) - (x > y); /* descending */
}

OPJ_BOOL opj_metal_t1_dec_prepare(opj_metal_t1_dec_t* b,
                                  const opj_metal_t1_dec_cblk_in* in,
                                  OPJ_UINT32 count, OPJ_BYTE** in_ptrs,
                                  OPJ_INT32** out_ptrs)
{
    OPJMetalT1DecShared* s = g_shared;
    OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
    const double t0 = opj_mtl_now();
    size_t nin = 0, nout = 0, nscratch = 0;
    OPJ_UINT32 i, ngroups, nthreads;
    id<MTLDevice> dev = s.device;
    id<MTLBuffer> buf;

    b->count = 0;
    b->nthreads = 0;
    for (i = 0; i < count; ++i) {
        const opj_metal_t1_dec_cblk_in* c = &in[i];
        if (c->w == 0 || c->h == 0 || c->w > OPJ_METAL_T1_DEC_MAX_SIDE ||
                c->h > OPJ_METAL_T1_DEC_MAX_SIDE ||
                c->w * c->h > OPJ_METAL_T1_DEC_MAX_AREA ||
                c->numbps > OPJ_METAL_T1_DEC_MAX_NUMBPS ||
                c->npasses > 0xffffffu || c->qmfbid > 1 || c->orient > 3 ||
                c->plane < -1 || c->plane >= (OPJ_INT32)b->nplanes) {
            return OPJ_FALSE;
        }
        nin += (c->len + 2u + 3u) & ~3u;
        if (c->plane < 0) {
            nout += (size_t)c->w * c->h;
        }
        nscratch += opj_mtl_dec_flags_size(c->w, c->h);
    }
    if (nin > 0xfffffff0u || nout > 0x3ffffff0u || nscratch > 0x3ffffff0u) {
        return OPJ_FALSE;
    }
    /* Dispatch order: sorted by coded length, descending, and dealt to the */
    /* SIMD-groups round-robin, so that every SIMD-group gets one of the     */
    /* longest code-blocks, rather than the longest ones sharing one         */
    /* SIMD-group, which would then take as long as all of them together.   */
    ngroups = (count + OPJ_MTL_DEC_SIMD - 1) / OPJ_MTL_DEC_SIMD;
    nthreads = ngroups * OPJ_MTL_DEC_SIMD;
    if (b->keys_cap < count) {
        uint64_t* p = (uint64_t*)realloc(b->keys, (size_t)count * sizeof(*p));
        if (!p) {
            return OPJ_FALSE;
        }
        b->keys = p;
        b->keys_cap = count;
    }
    @autoreleasepool {
#define OPJ_MTL_RESERVE(field, size) \
        buf = o.field; \
        if (!opj_mtl_reserve(dev, &buf, (size))) { \
            return OPJ_FALSE; \
        } \
        o.field = buf;
        OPJ_MTL_RESERVE(descBuf, (size_t)count * sizeof(opj_mtl_dec_desc));
        OPJ_MTL_RESERVE(inBuf, nin);
        OPJ_MTL_RESERVE(outBuf, nout * 4);
        OPJ_MTL_RESERVE(scratchBuf, nscratch * 4);
        OPJ_MTL_RESERVE(statusBuf, (size_t)count * 4);
        OPJ_MTL_RESERVE(orderBuf, (size_t)nthreads * 4);
#undef OPJ_MTL_RESERVE
    }

    opj_mtl_dec_desc* desc = (opj_mtl_dec_desc*)o.descBuf.contents;
    uint8_t* inb = (uint8_t*)o.inBuf.contents;
    int32_t* outb = (int32_t*)o.outBuf.contents;
    uint32_t* order = (uint32_t*)o.orderBuf.contents;
    uint32_t ioff = 0, ooff = 0, soff = 0;
    for (i = 0; i < count; ++i) {
        const opj_metal_t1_dec_cblk_in* c = &in[i];
        const uint32_t padded = (c->len + 2u + 3u) & ~3u;
        uint32_t hs;
        memset(&desc[i], 0, sizeof(desc[i]));
        desc[i].in_off = ioff;
        desc[i].len = c->len;
        desc[i].w_h = c->w | (c->h << 16);
        desc[i].orient_qmf = c->orient | (c->qmfbid << 8) |
                             ((uint32_t)(c->plane + 1) << 16);
        desc[i].bps_passes = c->numbps | (c->npasses << 8);
        memcpy(&hs, &c->half_step, sizeof(hs));
        desc[i].half_step = hs;
        /* the end-of-data marker of opj_mqc_init_dec(), and zero padding */
        memset(inb + ioff + c->len, 0, padded - c->len);
        inb[ioff + c->len] = 0xff;
        inb[ioff + c->len + 1] = 0xff;
        in_ptrs[i] = inb + ioff;
        ioff += padded;
        if (c->plane < 0) {
            desc[i].out_off = ooff;
            desc[i].out_stride = c->w;
            out_ptrs[i] = outb + ooff;
            ooff += c->w * c->h;
        } else {
            desc[i].out_off = c->out_off;
            desc[i].out_stride = c->out_stride;
            out_ptrs[i] = NULL;
        }
        desc[i].scratch_off = soff;
        soff += opj_mtl_dec_flags_size(c->w, c->h);
        b->keys[i] = ((uint64_t)c->len << 32) | i;
    }
    qsort(b->keys, count, sizeof(uint64_t), opj_mtl_dec_cmp_key);
    for (i = 0; i < nthreads; ++i) {
        order[i] = OPJ_MTL_DEC_HOLE;
    }
    for (i = 0; i < count; ++i) {
        /* rank i goes to lane i / ngroups of SIMD-group i % ngroups */
        order[(i % ngroups) * OPJ_MTL_DEC_SIMD + i / ngroups] =
            (uint32_t)(b->keys[i] & 0xffffffffu);
    }
    b->count = count;
    b->nthreads = nthreads;
    b->scratch_bytes = nscratch * 4;
    if (g_poison) {
        memset(o.outBuf.contents, 0xa5, o.outBuf.length);
        memset(o.statusBuf.contents, 0xa5, o.statusBuf.length);
        memset(o.scratchBuf.contents, 0xa5, o.scratchBuf.length);
    }
    b->prep_s = opj_mtl_now() - t0;
    return OPJ_TRUE;
}

OPJ_BOOL opj_metal_t1_dec_commit(opj_metal_t1_dec_t* b)
{
    OPJMetalT1DecShared* s = g_shared;
    const uint32_t nthreads = b->nthreads;

    b->t_commit = opj_mtl_now();
    b->gpu_s = 0;
    if (b->count == 0) {
        return OPJ_TRUE;
    }
    @autoreleasepool {
        OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
        id<MTLCommandBuffer> cb;
        id<MTLBlitCommandEncoder> blit;
        id<MTLComputeCommandEncoder> enc;
        if (!o.queue) {
            o.queue = [s.device newCommandQueue];
        }
        cb = [o.queue ?: s.queue commandBufferWithUnretainedReferences];
        blit = [cb blitCommandEncoder];
        NSUInteger p;
        if (!cb || !blit) {
            return OPJ_FALSE;
        }
        /* The kernel relies on zeroed flags */
        [blit fillBuffer:o.scratchBuf range:NSMakeRange(0, b->scratch_bytes)
                   value:0];
        [blit endEncoding];
        enc = [cb computeCommandEncoder];
        if (!enc) {
            return OPJ_FALSE;
        }
        [enc setComputePipelineState:s.pso];
        [enc setBuffer:o.descBuf offset:0 atIndex:0];
        [enc setBuffer:o.inBuf offset:0 atIndex:1];
        [enc setBuffer:o.outBuf offset:0 atIndex:2];
        for (p = 0; p < OPJ_METAL_T1_DEC_MAX_PLANES; ++p) {
            /* unused plane slots get a placeholder */
            id<MTLBuffer> pb = p < b->nplanes ? o.planes[p] : o.outBuf;
            [enc setBuffer:pb offset:0 atIndex:3 + p];
        }
        [enc setBuffer:o.scratchBuf offset:0 atIndex:7];
        [enc setBuffer:o.statusBuf offset:0 atIndex:8];
        [enc setBuffer:o.orderBuf offset:0 atIndex:9];
        [enc setBytes:&nthreads length:sizeof(nthreads) atIndex:10];
        [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(OPJ_MTL_DEC_TG, 1, 1)];
        [enc endEncoding];
        if (b->slot) {
            /* The GPU slot frees when the GPU is done, not when the */
            /* caller has finished its own share and released the batch */
            b->slot = 0;
            [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                   (void)done;
                   atomic_fetch_sub(&g_inflight, 1);
               }];
        }
        [cb commit];
        o.cb = cb;
    }
    return OPJ_TRUE;
}

OPJ_BOOL opj_metal_t1_dec_wait(opj_metal_t1_dec_t* b, OPJ_UINT32* status)
{
    const uint32_t count = b->count;
    uint32_t i, ngpu = 0;
    OPJ_BOOL ok = OPJ_TRUE;

    if (count == 0) {
        b->run_s = opj_mtl_now() - b->t_commit;
        return OPJ_TRUE;
    }
    @autoreleasepool {
        OPJMetalT1DecBatch* o = (__bridge OPJMetalT1DecBatch*)b->obj;
        const uint32_t* st = (const uint32_t*)o.statusBuf.contents;
        id<MTLCommandBuffer> cb = o.cb;
        o.cb = nil;
        if (!cb || !opj_mtl_wait(cb)) {
            ok = OPJ_FALSE;
        } else {
            b->gpu_s = cb.GPUEndTime - cb.GPUStartTime;
            if (g_profile) {
                const opj_mtl_dec_desc* desc =
                    (const opj_mtl_dec_desc*)o.descBuf.contents;
                uint64_t sum = 0;
                uint32_t maxlen = 0;
                for (i = 0; i < count; ++i) {
                    sum += desc[i].len;
                    maxlen = desc[i].len > maxlen ? desc[i].len : maxlen;
                }
                fprintf(stderr, "opj_metal_t1_dec: %u code-blocks, %llu bytes "
                        "(longest %u), %u planes, gpu %.2f ms\n", count,
                        (unsigned long long)sum, maxlen, b->nplanes,
                        1e3 * b->gpu_s);
            }
            for (i = 0; i < count; ++i) {
                status[i] = st[i];
                if (g_debug_refuse && i % g_debug_refuse == 0) {
                    status[i] = 1;
                }
                if (status[i] == 0) {
                    ngpu++;
                }
            }
        }
    }
    b->run_s = opj_mtl_now() - b->t_commit;
    if (!ok) {
        return OPJ_FALSE;
    }
    if (g_debug_fail) {
        /* Test hook: report a failure after the GPU wrote its outputs */
        return OPJ_FALSE;
    }
    atomic_fetch_add(&g_gpu_cblks, ngpu);
    return OPJ_TRUE;
}
