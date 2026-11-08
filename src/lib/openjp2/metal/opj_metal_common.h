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
 * Helpers shared by the Metal tier-1 encoder (opj_metal_t1.m) and decoder
 * (opj_metal_t1_dec.m). Objective-C only, internal to the library.
 */

#ifndef OPJ_METAL_COMMON_H
#define OPJ_METAL_COMMON_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Monotonic time in seconds */
double opj_mtl_now(void);

/* Pipeline of kernel name in lib, nil (and a message on stderr) on failure */
id<MTLComputePipelineState> opj_mtl_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                        NSString* name);

/* Grows *buf to at least size bytes (shared storage). Exact size the first
 * time, 25 % headroom when it must grow; a buffer more than 4 times too big
 * (and above 1 MB) is replaced by an exact one. */
BOOL opj_mtl_reserve(id<MTLDevice> dev, id<MTLBuffer> __strong* buf,
                     size_t size);

/* Waits for cb; NO (and a message on stderr) if it did not complete */
BOOL opj_mtl_wait(id<MTLCommandBuffer> cb);

#endif /* OPJ_METAL_COMMON_H */
