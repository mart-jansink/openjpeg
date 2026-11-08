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


/* Compiled with -fobjc-arc. See opj_metal_common.h. */

#include <stdio.h>
#include <mach/mach_time.h>

#include "opj_metal_common.h"

double opj_mtl_now(void)
{
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

id<MTLComputePipelineState> opj_mtl_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                        NSString* name)
{
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "opj_metal: kernel %s not found\n", name.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "opj_metal: pipeline %s: %s\n", name.UTF8String,
                err.localizedDescription.UTF8String);
    }
    return pso;
}

BOOL opj_mtl_reserve(id<MTLDevice> dev, id<MTLBuffer> __strong* buf,
                     size_t size)
{
    if (size == 0) {
        size = 16;
    }
    if (*buf && (*buf).length >= size &&
            !((*buf).length > 4 * size && (*buf).length > (1u << 20))) {
        return YES;
    }
    size_t cap = (*buf && (*buf).length < size) ? size + size / 4 : size;
    *buf = nil; /* release the old one first */
    *buf = [dev newBufferWithLength:cap options:MTLResourceStorageModeShared];
    return *buf != nil;
}

BOOL opj_mtl_wait(id<MTLCommandBuffer> cb)
{
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "opj_metal: command buffer failed: %s\n",
                cb.error.localizedDescription.UTF8String);
        return NO;
    }
    return YES;
}
