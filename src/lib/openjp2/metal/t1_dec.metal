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
 * Metal JPEG 2000 Tier-1 decoder kernel (code-block style 0), see
 * opj_metal_t1_dec.h. The CPU tables of t1_luts.h are pasted in at the
 * OPJ_T1_LUTS marker below by opj_metal_embed.cmake.
 *
 * One code-block per thread: the MQ decoder is serial, and so is the
 * context modelling, since every decoded bit changes the contexts of the
 * coefficients after it. Kernel t1_dec_cblk is a straight port of
 * opj_t1_decode_cblk() with the CPU flags layout (T1_SIGMA/CHI/MU/PI), the
 * flags in device memory, the MQ contexts and the tables in threadgroup
 * memory (they are on the serial critical path of every symbol).
 *
 * (Two alternatives measured slower on an M1 Max and were dropped: the
 * coefficient state as 32-bit column masks in threadgroup memory, and a
 * state machine around a single MQ decoding site to keep the lanes of a
 * SIMD-group converged.)
 *
 * The host zeroes the flags with a blit before the kernel runs; the kernel
 * writes every sample of its destination region, so that the CPU may
 * decode other code-blocks of the same planes meanwhile. It decodes into
 * the destination (a raster region of the batch or the tile plane itself)
 * and finishes with the dequantisation of opj_t1_clbl_decode_processor():
 * v / 2 for 5/3, (float)v * half_step for 9/7.
 */

#include <metal_stdlib>
using namespace metal;

#define OPJ_BYTE uchar
#define OPJ_INT16 short
#define T1_NMSEDEC_BITS 7
#define static constant
/*@OPJ_T1_LUTS@*/
#undef static

/* Must match opj_metal_t1_dec.m */
struct dec_desc {
    uint in_off;      /* byte offset of the coded bytes in in_buf */
    uint len;         /* coded bytes; in_buf[in_off + len ..+1] = 0xFF 0xFF */
    uint w_h;         /* w | (h << 16) */
    uint orient_qmf;  /* orient | (qmfbid << 8) | (dst << 16): dst 0 is the */
                      /* batch output, 1 + p the mapped plane p            */
    uint bps_passes;  /* numbps | (npasses << 8) */
    uint out_off;     /* int32 index of the top-left sample in dst */
    uint out_stride;  /* samples per row in dst */
    uint half_step;   /* float bits of 0.5f * band->stepsize */
    uint scratch_off; /* serial kernel: uint index of the flags */
    uint pad0;
    uint pad1;
    uint pad2;
};

/* MQ coder states of mqc.c: qe | nmps << 16 | nlps << 24, mps = index & 1 */
constant uint mqc_tab[94] = {
    0x03025601u, 0x02035601u, 0x0c043401u, 0x0d053401u, 0x12061801u, 0x13071801u,
    0x18080ac1u, 0x19090ac1u, 0x3a0a0521u, 0x3b0b0521u, 0x424c0221u, 0x434d0221u,
    0x0d0e5601u, 0x0c0f5601u, 0x1c105401u, 0x1d115401u, 0x1c124801u, 0x1d134801u,
    0x1c143801u, 0x1d153801u, 0x22163001u, 0x23173001u, 0x24182401u, 0x25192401u,
    0x281a1c01u, 0x291b1c01u, 0x2a3a1601u, 0x2b3b1601u, 0x1d1e5601u, 0x1c1f5601u,
    0x1c205401u, 0x1d215401u, 0x1e225101u, 0x1f235101u, 0x20244801u, 0x21254801u,
    0x22263801u, 0x23273801u, 0x24283401u, 0x25293401u, 0x262a3001u, 0x272b3001u,
    0x262c2801u, 0x272d2801u, 0x282e2401u, 0x292f2401u, 0x2a302201u, 0x2b312201u,
    0x2c321c01u, 0x2d331c01u, 0x2e341801u, 0x2f351801u, 0x30361601u, 0x31371601u,
    0x32381401u, 0x33391401u, 0x343a1201u, 0x353b1201u, 0x363c1101u, 0x373d1101u,
    0x383e0ac1u, 0x393f0ac1u, 0x3a4009c1u, 0x3b4109c1u, 0x3c4208a1u, 0x3d4308a1u,
    0x3e440521u, 0x3f450521u, 0x40460441u, 0x41470441u, 0x424802a1u, 0x434902a1u,
    0x444a0221u, 0x454b0221u, 0x464c0141u, 0x474d0141u, 0x484e0111u, 0x494f0111u,
    0x4a500085u, 0x4b510085u, 0x4c520049u, 0x4d530049u, 0x4e540025u, 0x4f550025u,
    0x50560015u, 0x51570015u, 0x52580009u, 0x53590009u, 0x545a0005u, 0x555b0005u,
    0x565a0001u, 0x575b0001u, 0x5c5c5601u, 0x5d5d5601u,
};

/* Context numbers (t1.h) */
#define CTX_ZC  0
#define CTX_SC  9
#define CTX_MAG 14
#define CTX_AGG 17
#define CTX_UNI 18
#define NUMCTXS 19

/* ------------------------------------------------------------------------ */
/* MQ decoder (mqc.c / mqc_inl.h, ISO 15444-1 C.3). The coded bytes are     */
/* followed by 0xFF 0xFF, as opj_mqc_init_dec() arranges, so bytein never   */
/* reads past len + 1.                                                      */
/* ------------------------------------------------------------------------ */

struct mq_dec {
    uint a;
    uint c;
    uint ct;
    uint bp;   /* index of the current byte in d */
};

static inline void mq_bytein(thread mq_dec& m, device const uchar* d)
{
    const uint l_c = d[m.bp + 1];
    if (d[m.bp] == 0xffu) {
        if (l_c > 0x8fu) {
            m.c += 0xff00u;
            m.ct = 8;
        } else {
            m.bp++;
            m.c += l_c << 9;
            m.ct = 7;
        }
    } else {
        m.bp++;
        m.c += l_c << 8;
        m.ct = 8;
    }
}

static inline void mq_init(thread mq_dec& m, device const uchar* d, uint len)
{
    m.bp = 0;
    m.ct = 0;
    m.c = (len == 0) ? (0xffu << 16) : ((uint)d[0] << 16);
    mq_bytein(m, d);
    m.c <<= 7;
    m.ct -= 7;
    m.a = 0x8000u;
}

static inline void mq_renormd(thread mq_dec& m, device const uchar* d)
{
    do {
        if (m.ct == 0) {
            mq_bytein(m, d);
        }
        m.a <<= 1;
        m.c <<= 1;
        m.ct--;
    } while (m.a < 0x8000u);
}

#define DEC_TG 64   /* threads per threadgroup of t1_dec_cblk */

/* mqc_tab repacked so that a context can hold its whole state: */
/* qe | mps << 16 | nmps << 17 | nlps << 24 (entry i has mps = i & 1) */
constant uint mqc_pk[94] = {
    0x03045601u, 0x02075601u, 0x0c083401u, 0x0d0b3401u, 0x120c1801u, 0x130f1801u,
    0x18100ac1u, 0x19130ac1u, 0x3a140521u, 0x3b170521u, 0x42980221u, 0x439b0221u,
    0x0d1c5601u, 0x0c1f5601u, 0x1c205401u, 0x1d235401u, 0x1c244801u, 0x1d274801u,
    0x1c283801u, 0x1d2b3801u, 0x222c3001u, 0x232f3001u, 0x24302401u, 0x25332401u,
    0x28341c01u, 0x29371c01u, 0x2a741601u, 0x2b771601u, 0x1d3c5601u, 0x1c3f5601u,
    0x1c405401u, 0x1d435401u, 0x1e445101u, 0x1f475101u, 0x20484801u, 0x214b4801u,
    0x224c3801u, 0x234f3801u, 0x24503401u, 0x25533401u, 0x26543001u, 0x27573001u,
    0x26582801u, 0x275b2801u, 0x285c2401u, 0x295f2401u, 0x2a602201u, 0x2b632201u,
    0x2c641c01u, 0x2d671c01u, 0x2e681801u, 0x2f6b1801u, 0x306c1601u, 0x316f1601u,
    0x32701401u, 0x33731401u, 0x34741201u, 0x35771201u, 0x36781101u, 0x377b1101u,
    0x387c0ac1u, 0x397f0ac1u, 0x3a8009c1u, 0x3b8309c1u, 0x3c8408a1u, 0x3d8708a1u,
    0x3e880521u, 0x3f8b0521u, 0x408c0441u, 0x418f0441u, 0x429002a1u, 0x439302a1u,
    0x44940221u, 0x45970221u, 0x46980141u, 0x479b0141u, 0x489c0111u, 0x499f0111u,
    0x4aa00085u, 0x4ba30085u, 0x4ca40049u, 0x4da70049u, 0x4ea80025u, 0x4fab0025u,
    0x50ac0015u, 0x51af0015u, 0x52b00009u, 0x53b30009u, 0x54b40005u, 0x55b70005u,
    0x56b40001u, 0x57b70001u, 0x5cb85601u, 0x5dbb5601u,
};

/* The MQ contexts of one thread, interleaved with those of the other */
/* threads of its threadgroup: context i at p[i * cs] */
struct ctx_ref {
    threadgroup uint* p;
    uint cs;
};

/* mq_decode() with the contexts as mqc_pk words: the usual path (MPS, no */
/* renormalisation) is a single threadgroup load. pk is mqc_pk in         */
/* threadgroup memory. */

static inline uint mq_decode_pk(thread mq_dec& m, device const uchar* d,
                                ctx_ref cx, uint ctxno,
                                threadgroup const uint* pk)
{
    threadgroup uint* c = cx.p + ctxno * cx.cs;
    const uint w = *c;
    const uint qe = w & 0xffffu;
    const uint mps = (w >> 16) & 1u;
    uint v;
    m.a -= qe;
    if ((m.c >> 16) < qe) {
        /* LPS exchange */
        if (m.a < qe) {
            v = mps;
            *c = pk[(w >> 17) & 0x7fu];
        } else {
            v = mps ^ 1u;
            *c = pk[w >> 24];
        }
        m.a = qe;
        mq_renormd(m, d);
    } else {
        m.c -= qe << 16;
        if ((m.a & 0x8000u) == 0) {
            /* MPS exchange */
            if (m.a < qe) {
                v = mps ^ 1u;
                *c = pk[w >> 24];
            } else {
                v = mps;
                *c = pk[(w >> 17) & 0x7fu];
            }
            mq_renormd(m, d);
        } else {
            v = mps;
        }
    }
    return v;
}

/* Destination of a code-block */
static inline device int* dec_dst(uint sel, device int* out,
                                  device int* p0, device int* p1,
                                  device int* p2, device int* p3)
{
    return sel == 0 ? out : sel == 1 ? p0 : sel == 2 ? p1 : sel == 3 ? p2 : p3;
}


/* ======================================================================== */
/* t1_dec_cblk: port of t1.c with its flags.                                */
/* ======================================================================== */

#define T1_SIGMA_0  (1U << 0)
#define T1_SIGMA_1  (1U << 1)
#define T1_SIGMA_2  (1U << 2)
#define T1_SIGMA_3  (1U << 3)
#define T1_SIGMA_4  (1U << 4)
#define T1_SIGMA_5  (1U << 5)
#define T1_SIGMA_6  (1U << 6)
#define T1_SIGMA_7  (1U << 7)
#define T1_SIGMA_8  (1U << 8)
#define T1_SIGMA_15 (1U << 15)
#define T1_SIGMA_16 (1U << 16)
#define T1_SIGMA_17 (1U << 17)
#define T1_CHI_0_I  18
#define T1_CHI_1_I  19
#define T1_MU_0     (1U << 20)
#define T1_PI_0     (1U << 21)
#define T1_CHI_2_I  22
#define T1_PI_1     (1U << 24)
#define T1_PI_2     (1U << 27)
#define T1_PI_3     (1U << 30)
#define T1_CHI_5_I  31
#define T1_SIGMA_NEIGHBOURS (T1_SIGMA_0 | T1_SIGMA_1 | T1_SIGMA_2 | T1_SIGMA_3 | T1_SIGMA_5 | T1_SIGMA_6 | T1_SIGMA_7 | T1_SIGMA_8)
#define T1_SIGMA_THIS T1_SIGMA_4
#define T1_MU_THIS    T1_MU_0
#define T1_PI_THIS    T1_PI_0
#define T1_PI_ALL     (T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3)

/* opj_t1_getctxtno_sc_or_spb_index() */
static inline uint sc_index(uint fX, uint pfX, uint nfX, uint ci)
{
    uint lu = (fX >> (ci * 3U)) & (T1_SIGMA_1 | T1_SIGMA_3 | T1_SIGMA_5 |
                                   T1_SIGMA_7);
    lu |= (pfX >> (T1_CHI_1_I + (ci * 3U))) & (1U << 0);
    lu |= (nfX >> (T1_CHI_1_I - 2U + (ci * 3U))) & (1U << 2);
    if (ci == 0U) {
        lu |= (fX >> (T1_CHI_0_I - 4U)) & (1U << 4);
    } else {
        lu |= (fX >> (T1_CHI_1_I - 4U + ((ci - 1U) * 3U))) & (1U << 4);
    }
    lu |= (fX >> (T1_CHI_2_I - 6U + (ci * 3U))) & (1U << 6);
    return lu;
}

/* opj_t1_update_flags_macro() without VSC; the flags of the coefficient  */
/* itself are in the register 'flags', its neighbours' in memory at f.    */
static inline void update_flags(thread uint& flags, device uint* f, uint ci,
                                uint s, uint stride)
{
    f[-1] |= T1_SIGMA_5 << (3U * ci);
    flags |= ((s << T1_CHI_1_I) | T1_SIGMA_4) << (3U * ci);
    f[1] |= T1_SIGMA_3 << (3U * ci);
    if (ci == 0U) {
        device uint* north = f - stride;
        *north |= (s << T1_CHI_5_I) | T1_SIGMA_16;
        north[-1] |= T1_SIGMA_17;
        north[1] |= T1_SIGMA_15;
    }
    if (ci == 3U) {
        device uint* south = f + stride;
        *south |= (s << T1_CHI_0_I) | T1_SIGMA_1;
        south[-1] |= T1_SIGMA_2;
        south[1] |= T1_SIGMA_0;
    }
}

/* Significance decoding of coefficient ci of the column at f (both the    */
/* sigpass step, with the zero coding, and the cleanup step).             */
static inline void dec_sig(thread mq_dec& m, device const uchar* d,
                           ctx_ref cx, threadgroup const uint* pk,
                           threadgroup const uchar* lut_zc,
                           threadgroup const uchar* lut_sc,
                           thread uint& flags, device uint* f,
                           device int* dp, uint ci, uint stride,
                           int oneplushalf, bool zc)
{
    if (zc) {
        const uint ctxt1 = lut_zc[(flags >> (ci * 3U)) & T1_SIGMA_NEIGHBOURS];
        if (!mq_decode_pk(m, d, cx, ctxt1, pk)) {
            return;
        }
    }
    const uint lu = sc_index(flags, f[-1], f[1], ci);
    /* lut_sc: lut_ctxno_sc | lut_spb << 7 */
    const uint sc = lut_sc[lu];
    const uint v = mq_decode_pk(m, d, cx, sc & 0x7fu, pk) ^ (sc >> 7);
    *dp = v ? -oneplushalf : oneplushalf;
    update_flags(flags, f, ci, v, stride);
}

/* Decodes code-block dsc, with its flags at flags_buf (zeroed by the host), */
/* into dst. Every sample of the w x h region of dst is written; only the  */
/* significant ones are read. */
static void dec_cblk(const dec_desc dsc, uint gid, device uint* status,
                     device const uchar* d, device int* dst,
                     device uint* flags_buf, ctx_ref cx,
                     threadgroup const uint* pk,
                     threadgroup const uchar* lut_zc,
                     threadgroup const uchar* lut_sc)
{
    const uint w = dsc.w_h & 0xffffu;
    const uint h = dsc.w_h >> 16;
    const uint qmfbid = (dsc.orient_qmf >> 8) & 0xffu;
    const uint numbps = dsc.bps_passes & 0xffu;
    const uint npasses = dsc.bps_passes >> 8;
    const uint ds = dsc.out_stride;

    status[gid] = 0;
    if (npasses == 0 || numbps == 0) {
        for (uint y = 0; y < h; ++y) {
            for (uint x = 0; x < w; ++x) {
                dst[y * ds + x] = 0;
            }
        }
        return;
    }

    /* Flags, as opj_t1_allocate_buffers() */
    const uint stride = w + 2U;
    const uint fh = (h + 3U) / 4U;
    for (uint x = 0; x < stride; ++x) {
        flags_buf[x] = T1_PI_ALL;
        flags_buf[(fh + 1) * stride + x] = T1_PI_ALL;
    }
    if (h % 4) {
        uint v = 0;
        if (h % 4 == 1) {
            v = T1_PI_1 | T1_PI_2 | T1_PI_3;
        } else if (h % 4 == 2) {
            v = T1_PI_2 | T1_PI_3;
        } else {
            v = T1_PI_3;
        }
        for (uint x = 0; x < stride; ++x) {
            flags_buf[fh * stride + x] = v;
        }
    }
    device uint* f0 = flags_buf + 1 + stride; /* T1_FLAGS(0, 0) */
    const uint hfull = h & ~3U;

    mq_dec m;
    for (uint i = 0; i < NUMCTXS; ++i) {
        cx.p[i * cx.cs] = pk[0];
    }
    cx.p[CTX_UNI * cx.cs] = pk[46 * 2];
    cx.p[CTX_AGG * cx.cs] = pk[3 * 2];
    cx.p[CTX_ZC * cx.cs] = pk[4 * 2];
    mq_init(m, d, dsc.len);

    int bpno = (int)numbps;  /* bpno_plus_one of opj_t1_decode_cblk() */
    uint passtype = 2;
    for (uint passno = 0; passno < npasses && bpno >= 1; ++passno) {
        const int one = 1 << bpno;
        const int halfv = one >> 1;
        const int oneplushalf = one | halfv;
        if (passtype == 0) {
            /* ---- significance propagation pass ---- */
            for (uint k = 0; k < hfull; k += 4) {
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    uint flags = *f;
                    if (flags == 0) {
                        continue;
                    }
                    for (uint ci = 0; ci < 4; ++ci) {
                        if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U))) == 0U &&
                                (flags & (T1_SIGMA_NEIGHBOURS << (ci * 3U))) != 0U) {
                            dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f,
                                    dst + (k + ci) * ds + i, ci, stride,
                                    oneplushalf, true);
                            flags |= T1_PI_THIS << (ci * 3U);
                        }
                    }
                    *f = flags;
                }
            }
            if (hfull < h) {
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (hfull / 4) * stride + i;
                    uint flags = *f;
                    for (uint ci = 0; ci < h - hfull; ++ci) {
                        if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U))) == 0U &&
                                (flags & (T1_SIGMA_NEIGHBOURS << (ci * 3U))) != 0U) {
                            dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f,
                                    dst + (hfull + ci) * ds + i, ci, stride,
                                    oneplushalf, true);
                            flags |= T1_PI_THIS << (ci * 3U);
                        }
                    }
                    *f = flags;
                }
            }
        } else if (passtype == 1) {
            /* ---- magnitude refinement pass ---- */
            for (uint k = 0; k < h; k += 4) {
                const uint lim = min(4u, h - k);
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    uint flags = *f;
                    if (flags == 0) {
                        continue;
                    }
                    for (uint ci = 0; ci < lim; ++ci) {
                        if ((flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U))) ==
                                (T1_SIGMA_THIS << (ci * 3U))) {
                            const uint sf = flags >> (ci * 3U);
                            uint ctxt = (sf & T1_SIGMA_NEIGHBOURS) ? CTX_MAG + 1 : CTX_MAG;
                            ctxt = (sf & T1_MU_0) ? CTX_MAG + 2 : ctxt;
                            const uint v = mq_decode_pk(m, d, cx, ctxt, pk);
                            device int* dp = dst + (k + ci) * ds + i;
                            const int cur = *dp;
                            *dp = cur + (((v ^ (cur < 0 ? 1u : 0u)) != 0) ? halfv : -halfv);
                            flags |= T1_MU_THIS << (ci * 3U);
                        }
                    }
                    *f = flags;
                }
            }
        } else {
            /* ---- cleanup pass ---- */
            for (uint k = 0; k < hfull; k += 4) {
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    uint flags = *f;
                    device int* dp = dst + k * ds + i;
                    if (flags == 0) {
                        if (!mq_decode_pk(m, d, cx, CTX_AGG, pk)) {
                            continue;
                        }
                        uint runlen = mq_decode_pk(m, d, cx, CTX_UNI, pk);
                        runlen = (runlen << 1) | mq_decode_pk(m, d, cx, CTX_UNI, pk);
                        /* coefficient runlen is significant: no zero coding */
                        dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f, dp + runlen * ds,
                                runlen, stride, oneplushalf, false);
                        for (uint ci = runlen + 1; ci < 4; ++ci) {
                            dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f, dp + ci * ds, ci,
                                    stride, oneplushalf, true);
                        }
                    } else {
                        for (uint ci = 0; ci < 4; ++ci) {
                            if (!(flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U)))) {
                                dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f, dp + ci * ds, ci,
                                        stride, oneplushalf, true);
                            }
                        }
                    }
                    *f = flags & ~T1_PI_ALL;
                }
            }
            if (hfull < h) {
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (hfull / 4) * stride + i;
                    uint flags = *f;
                    for (uint ci = 0; ci < h - hfull; ++ci) {
                        if (!(flags & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U)))) {
                            dec_sig(m, d, cx, pk, lut_zc, lut_sc, flags, f,
                                    dst + (hfull + ci) * ds + i, ci, stride,
                                    oneplushalf, true);
                        }
                    }
                    *f = flags & ~T1_PI_ALL;
                }
            }
        }
        if (++passtype == 3) {
            passtype = 0;
            bpno--;
        }
    }

    /* Dequantisation of opj_t1_clbl_decode_processor(), in place: v / 2  */
    /* for 5/3, (float)v * half_step for 9/7. Only significant coefficients */
    /* were written; the others are 0, which maps to 0 (int and +0.0f).     */
    const float half_step = as_type<float>(dsc.half_step);
    for (uint k = 0; k < h; k += 4) {
        const uint lim = min(4u, h - k);
        for (uint i = 0; i < w; ++i) {
            const uint flags = f0[(k / 4) * stride + i];
            for (uint ci = 0; ci < lim; ++ci) {
                device int* dp = dst + (k + ci) * ds + i;
                if (flags & (T1_SIGMA_THIS << (ci * 3U))) {
                    *dp = qmfbid == 1 ? *dp / 2 :
                          as_type<int>((float)*dp * half_step);
                } else {
                    *dp = 0;
                }
            }
        }
    }
}

/* Copies the tables into threadgroup memory: shorter latency than constant */
/* memory on the serial critical path of every symbol */
static inline void load_tables(threadgroup uchar* tg_lut_zc,
                               threadgroup uchar* tg_lut_sc,
                               threadgroup uint* tg_pk, uint lid, uint ntg)
{
    for (uint i = lid; i < 4 * 512; i += ntg) {
        tg_lut_zc[i] = lut_ctxno_zc[i];
    }
    for (uint i = lid; i < 256; i += ntg) {
        tg_lut_sc[i] = (uchar)(lut_ctxno_sc[i] | (lut_spb[i] << 7));
    }
    for (uint i = lid; i < 94; i += ntg) {
        tg_pk[i] = mqc_pk[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* One code-block per thread, in the order of order[] (count entries,    */
/* holes included) */
kernel void t1_dec_cblk(device const dec_desc* descs [[buffer(0)]],
                        device const uchar* in_buf [[buffer(1)]],
                        device int* out_buf [[buffer(2)]],
                        device int* plane0 [[buffer(3)]],
                        device int* plane1 [[buffer(4)]],
                        device int* plane2 [[buffer(5)]],
                        device int* plane3 [[buffer(6)]],
                        device uint* scratch [[buffer(7)]],
                        device uint* status [[buffer(8)]],
                        device const uint* order [[buffer(9)]],
                        constant uint& count [[buffer(10)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]])
{
    threadgroup uint tg_ctx[NUMCTXS * DEC_TG];
    threadgroup uchar tg_lut_zc[4 * 512];
    threadgroup uchar tg_lut_sc[256];
    threadgroup uint tg_pk[94];
    load_tables(tg_lut_zc, tg_lut_sc, tg_pk, lid, ntg);
    if (tid >= count) {
        return;
    }
    const uint gid = order[tid];
    if (gid == 0xffffffffu) {
        return; /* a hole of the interleaved dispatch order */
    }
    const dec_desc dsc = descs[gid];
    device const uchar* d = in_buf + dsc.in_off;
    device int* dst = dec_dst(dsc.orient_qmf >> 16, out_buf, plane0, plane1,
                              plane2, plane3) + dsc.out_off;
    threadgroup const uchar* lut_zc = tg_lut_zc + ((dsc.orient_qmf & 0xffu) << 9);
    const ctx_ref cx = { tg_ctx + lid, DEC_TG };
    dec_cblk(dsc, gid, status, d, dst, scratch + dsc.scratch_off, cx, tg_pk,
             lut_zc, tg_lut_sc);
}
