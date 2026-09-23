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
 * JPEG 2000 Tier-1 (EBCOT) encoder kernels, bit-exact with OpenJPEG's
 * opj_t1_encode_cblk() for cblksty == 0.
 *
 * This file is compiled at run time. At build time the content of
 * t1_luts.h is pasted at the OPJ_T1_LUTS marker below (see
 * opj_metal_embed.cmake), so the context and distortion tables are the
 * very same ones the CPU encoder uses.
 *
 * Kernel t1_enc_cblk_serial: design (a), one thread per code-block, a
 *   straight port of the CPU code (same flags layout) with the flags in
 *   device memory.
 * Kernel t1_enc_cblk_tg: design (b), one SIMD-group per code-block, see
 *   the comment in front of it.
 */

#include <metal_stdlib>
using namespace metal;

#define OPJ_BYTE uchar
#define OPJ_INT16 short
#define T1_NMSEDEC_BITS 7
#define static constant
/*@OPJ_T1_LUTS@*/
#undef static

#define T1_NMSEDEC_FRACBITS (T1_NMSEDEC_BITS-1)

/* Must match opj_metal_t1.m */
#define MAXPASSES 76

struct cblk_desc {
    uint data_off;    /* int32 offset of the raster w*h input (the region) */
    uint out_off;     /* design (a): byte offset in the output buffer;     */
                      /* out[out_off] is the fake byte, coded bytes start */
                      /* at out_off + 1                                   */
    uint scratch_off; /* design (a): uint offset of the flags scratch     */
    uint w_h;         /* w | (h << 16) */
    uint orient_sty;  /* orient | (cblksty << 8) */
    uint pass_off;    /* design (a): first pass record in the pass pool   */
    uint region;      /* design (b): bytes of the region at data_off      */
    uint pad;
};

/* status values */
#define ST_OK 0
#define ST_PASSES 1   /* more passes than MAXPASSES (cannot happen) */
#define ST_RETRY 2    /* pool full: input restored, retry on the GPU */
#define ST_BIG 3      /* left to t1_enc_cblk_serial */
#define ST_OUTFULL 4  /* coded bytes do not fit in the region: CPU */

struct cblk_res {
    uint numbps;
    uint totalpasses;
    uint nbytes;
    uint status;
    uint pass_off;    /* first pass record in the pass pool */
    uint region_end;  /* design (b): end of the symbols kept in the region */
    uint pad0;
    uint pad1;
};

/* Pass record. Design (b) context modelling stores the symbol count in */
/* rate and the symbol location in len (region offset, or pool offset   */
/* | LOC_POOL); the MQ kernel replaces them by rate and len.            */
struct pass_res {
    uint rate;
    uint len;
    int nmsedec;
};

#define LOC_POOL 0x80000000u

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
/* MQ encoder (mqc.c / mqc_inl.h). The byte at bp is kept in the register B  */
/* because it is the only byte a carry can still change; it is stored when  */
/* bp advances and at the end.                                               */
/* ------------------------------------------------------------------------ */

/* Output of the MQ coder written into the code-block region of design (b), */
/* behind the symbols not read yet: a byte may only be stored at region    */
/* index < lim (index 0 is the fake byte, so out[i] is region[i + 1]).     */
struct region_out {
    device uchar* p; /* region + 1 */
    uint lim;
    bool ovf;
};

static inline void mq_put(device uchar* o, int i, uint v)
{
    o[i] = (uchar)v;
}

static inline uint mq_get(device uchar* o, int i)
{
    return o[i];
}

static inline void mq_put(thread region_out* o, int i, uint v)
{
    if ((uint)(i + 1) < o->lim) {
        o->p[i] = (uchar)v;
    } else {
        o->ovf = true;
    }
}

static inline uint mq_get(thread region_out* o, int i)
{
    return o->p[i];
}

struct mq_enc {
    uint a;
    uint c;
    uint ct;
    int bp;     /* index relative to the start of the coded bytes; -1 = fake */
    uint B;     /* value of the byte at bp */
};

template <typename OUT>
static inline void mq_byteout(thread mq_enc& m, OUT out)
{
    if (m.B == 0xffu) {
        mq_put(out, m.bp, m.B);
        m.bp++;
        m.B = (m.c >> 20) & 0xffu;
        m.c &= 0xfffffu;
        m.ct = 7;
    } else if ((m.c & 0x8000000u) == 0) {
        mq_put(out, m.bp, m.B);
        m.bp++;
        m.B = (m.c >> 19) & 0xffu;
        m.c &= 0x7ffffu;
        m.ct = 8;
    } else {
        m.B++;
        if (m.B == 0xffu) {
            m.c &= 0x7ffffffu;
            mq_put(out, m.bp, m.B);
            m.bp++;
            m.B = (m.c >> 20) & 0xffu;
            m.c &= 0xfffffu;
            m.ct = 7;
        } else {
            mq_put(out, m.bp, m.B);
            m.bp++;
            m.B = (m.c >> 19) & 0xffu;
            m.c &= 0x7ffffu;
            m.ct = 8;
        }
    }
}

/* Renormalisation, done in chunks of up to ct bits instead of bit by bit */
template <typename OUT>
static inline void mq_renorm(thread mq_enc& m, OUT out)
{
    uint n = clz(m.a) - 16u; /* a != 0, a < 0x8000 here */
    while (n != 0) {
        uint s = min(n, m.ct);
        m.a <<= s;
        m.c <<= s;
        m.ct -= s;
        n -= s;
        if (m.ct == 0) {
            mq_byteout(m, out);
        }
    }
}

template <typename OUT, typename CTXS>
static inline void mq_encode(thread mq_enc& m, CTXS ctxs, uint ctxno, uint d,
                             OUT out)
{
    uint st = ctxs[ctxno];
    uint e = mqc_tab[st];
    uint qe = e & 0xffffu;
    m.a -= qe;
    if ((st & 1u) == d) {
        /* MPS */
        if ((m.a & 0x8000u) == 0) {
            if (m.a < qe) {
                m.a = qe;
            } else {
                m.c += qe;
            }
            ctxs[ctxno] = (uchar)((e >> 16) & 0xffu);
            mq_renorm(m, out);
        } else {
            m.c += qe;
        }
    } else {
        /* LPS */
        if (m.a < qe) {
            m.c += qe;
        } else {
            m.a = qe;
        }
        ctxs[ctxno] = (uchar)(e >> 24);
        mq_renorm(m, out);
    }
}

/* opj_mqc_flush(); returns nothing, m.bp is then opj_mqc_numbytes() */
template <typename OUT>
static inline void mq_flush(thread mq_enc& m, OUT out)
{
    uint tempc = m.c + m.a;
    m.c |= 0xffffu;
    if (m.c >= tempc) {
        m.c -= 0x8000u;
    }
    m.c <<= m.ct;
    mq_byteout(m, out);
    m.c <<= m.ct;
    mq_byteout(m, out);
    mq_put(out, m.bp, m.B);
    if (m.B != 0xffu) {
        m.bp++;
    }
}

template <typename CTXS>
static inline void mq_init(thread mq_enc& m, CTXS ctxs)
{
    for (uint i = 0; i < NUMCTXS; ++i) {
        ctxs[i] = 0;
    }
    ctxs[CTX_UNI] = 46 * 2;
    ctxs[CTX_AGG] = 3 * 2;
    ctxs[CTX_ZC] = 4 * 2;
    m.a = 0x8000u;
    m.c = 0;
    m.ct = 12;
    m.bp = -1;
    m.B = 0; /* the fake byte */
}

static inline short nmsedec_sig(uint x, uint bitpos)
{
    if (bitpos > 0) {
        return lut_nmsedec_sig[(x >> bitpos) & ((1u << T1_NMSEDEC_BITS) - 1)];
    }
    return lut_nmsedec_sig0[x & ((1u << T1_NMSEDEC_BITS) - 1)];
}

static inline short nmsedec_ref(uint x, uint bitpos)
{
    if (bitpos > 0) {
        return lut_nmsedec_ref[(x >> bitpos) & ((1u << T1_NMSEDEC_BITS) - 1)];
    }
    return lut_nmsedec_ref0[x & ((1u << T1_NMSEDEC_BITS) - 1)];
}

/* Rate bookkeeping at the end of opj_t1_encode_cblk() */
template <typename OUT>
static inline void finish_rates(device pass_res* pr, uint npasses,
                                uint numbytes, OUT out)
{
    uint last = numbytes;
    for (uint p = npasses; p > 0;) {
        --p;
        if (pr[p].rate > last) {
            pr[p].rate = last;
        } else {
            last = pr[p].rate;
        }
    }
    uint prev = 0;
    for (uint p = 0; p < npasses; ++p) {
        uint r = pr[p].rate;
        if (mq_get(out, (int)r - 1) == 0xff) {
            r--;
        }
        pr[p].rate = r;
        pr[p].len = r - prev;
        prev = r;
    }
}

/* ======================================================================== */
/* Design (a): one thread per code-block, port of t1.c with its flags.     */
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
#define T1_SIGMA_10 (1U << 10)
#define T1_SIGMA_13 (1U << 13)
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

static inline void update_flags(device uint* f, uint ci, uint s, uint stride)
{
    f[-1] |= T1_SIGMA_5 << (3U * ci);
    f[0] |= ((s << T1_CHI_1_I) | T1_SIGMA_4) << (3U * ci);
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

kernel void t1_enc_cblk_serial(device const cblk_desc* descs [[buffer(0)]],
                               device uint* data_buf [[buffer(1)]],
                               device uchar* out_buf [[buffer(2)]],
                               device uint* scratch [[buffer(3)]],
                               device cblk_res* res_buf [[buffer(4)]],
                               device pass_res* pass_buf [[buffer(5)]],
                               constant uint& count [[buffer(6)]],
                               device const uint* idx [[buffer(7)]],
                               constant uint& use_idx [[buffer(8)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid >= count) {
        return;
    }
    const uint gid = use_idx ? idx[tid] : tid;
    const cblk_desc d = descs[gid];
    const uint w = d.w_h & 0xffffu;
    const uint h = d.w_h >> 16;
    const uint orient = d.orient_sty & 0xffu;
    device uint* data = data_buf + d.data_off;
    device uchar* out = out_buf + d.out_off + 1; /* out[-1] = fake byte */
    device cblk_res* res = res_buf + gid;
    device pass_res* pr = pass_buf + d.pass_off;
    res->pass_off = d.pass_off;
    constant uchar* lut_zc = lut_ctxno_zc + (orient << 9);

    /* Sign-magnitude conversion (in place) and max magnitude */
    int maxv = 0;
    for (uint i = 0; i < w * h; ++i) {
        int t = (int)data[i];
        if (t < 0) {
            if (t == INT_MIN) {
                t = INT_MIN + 1;
            }
            maxv = max(maxv, -t);
            data[i] = (uint)(-t) | 0x80000000u;
        } else {
            maxv = max(maxv, t);
        }
    }
    uint numbps = maxv ? (uint)((int)(31 - clz((uint)maxv)) + 1 -
                                T1_NMSEDEC_FRACBITS) : 0u;
    res->numbps = numbps;
    res->status = 0;
    out[-1] = 0;
    if (numbps == 0 || (int)numbps - 1 < 0) {
        res->totalpasses = 0;
        res->nbytes = 0;
        return;
    }

    /* Flags, as opj_t1_allocate_buffers() */
    const uint stride = w + 2U;
    const uint fh = (h + 3U) / 4U;
    device uint* flags = scratch + d.scratch_off;
    for (uint i = 0; i < stride * (fh + 2U); ++i) {
        flags[i] = 0;
    }
    for (uint x = 0; x < stride; ++x) {
        flags[x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
        flags[(fh + 1) * stride + x] = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
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
            flags[fh * stride + x] = v;
        }
    }
    device uint* f0 = flags + 1 + stride; /* T1_FLAGS(0, 0) */

    mq_enc m;
    uchar ctxs[NUMCTXS];
    mq_init(m, ctxs);

    int bpno = (int)numbps - 1;
    uint passtype = 2;
    uint passno = 0;
    for (; bpno >= 0; ++passno) {
        const uint one = 1u << (bpno + T1_NMSEDEC_FRACBITS);
        int nmsedec = 0;
        if (passtype == 0) {
            /* ---- significance propagation pass ---- */
            for (uint k = 0; k < h; k += 4) {
                const uint lim = min(4u, h - k);
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    if (*f == 0U) {
                        continue;
                    }
                    for (uint ci = 0; ci < lim; ++ci) {
                        const uint flg = *f;
                        if ((flg & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U))) == 0U &&
                                (flg & (T1_SIGMA_NEIGHBOURS << (ci * 3U))) != 0U) {
                            const uint dv = data[(k + ci) * w + i];
                            const uint ctxt1 = lut_zc[(flg >> (ci * 3U)) & T1_SIGMA_NEIGHBOURS];
                            uint v = (dv & one) ? 1u : 0u;
                            mq_encode(m, ctxs, ctxt1, v, out);
                            if (v) {
                                const uint lu = sc_index(*f, f[-1], f[1], ci);
                                v = dv >> 31;
                                nmsedec += nmsedec_sig(dv & 0x7fffffffu, (uint)bpno);
                                mq_encode(m, ctxs, lut_ctxno_sc[lu], v ^ lut_spb[lu], out);
                                update_flags(f, ci, v, stride);
                            }
                            *f |= T1_PI_THIS << (ci * 3U);
                        }
                    }
                }
            }
        } else if (passtype == 1) {
            /* ---- magnitude refinement pass ---- */
            for (uint k = 0; k < h; k += 4) {
                const uint lim = min(4u, h - k);
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    const uint flg = *f;
                    uint upd = flg;
                    if ((flg & (T1_SIGMA_4 | T1_SIGMA_7 | T1_SIGMA_10 | T1_SIGMA_13)) == 0) {
                        continue;
                    }
                    if (lim == 4 && (flg & (T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3)) ==
                            (T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3)) {
                        continue;
                    }
                    for (uint ci = 0; ci < lim; ++ci) {
                        if ((flg & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U))) ==
                                (T1_SIGMA_THIS << (ci * 3U))) {
                            const uint sf = flg >> (ci * 3U);
                            uint ctxt = (sf & T1_SIGMA_NEIGHBOURS) ? CTX_MAG + 1 : CTX_MAG;
                            ctxt = (sf & T1_MU_0) ? CTX_MAG + 2 : ctxt;
                            const uint ab = data[(k + ci) * w + i] & 0x7fffffffu;
                            nmsedec += nmsedec_ref(ab, (uint)bpno);
                            mq_encode(m, ctxs, ctxt, (ab & one) ? 1u : 0u, out);
                            upd |= T1_MU_THIS << (ci * 3U);
                        }
                    }
                    *f = upd;
                }
            }
        } else {
            /* ---- cleanup pass ---- */
            for (uint k = 0; k < h; k += 4) {
                const uint lim = min(4u, h - k);
                for (uint i = 0; i < w; ++i) {
                    device uint* f = f0 + (k / 4) * stride + i;
                    uint runlen = 0;
                    const uint agg = (lim == 4) && (*f == 0);
                    if (agg) {
                        for (runlen = 0; runlen < 4; ++runlen) {
                            if (data[(k + runlen) * w + i] & one) {
                                break;
                            }
                        }
                        mq_encode(m, ctxs, CTX_AGG, runlen != 4, out);
                        if (runlen == 4) {
                            continue;
                        }
                        mq_encode(m, ctxs, CTX_UNI, runlen >> 1, out);
                        mq_encode(m, ctxs, CTX_UNI, runlen & 1, out);
                    }
                    const uint check = (T1_SIGMA_4 | T1_SIGMA_7 | T1_SIGMA_10 | T1_SIGMA_13 |
                                        T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
                    if ((*f & check) == check) {
                        if (runlen == 0) {
                            *f &= ~(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
                        } else if (runlen == 1) {
                            *f &= ~(T1_PI_1 | T1_PI_2 | T1_PI_3);
                        } else if (runlen == 2) {
                            *f &= ~(T1_PI_2 | T1_PI_3);
                        } else if (runlen == 3) {
                            *f &= ~(T1_PI_3);
                        }
                        continue;
                    }
                    for (uint ci = runlen; ci < lim; ++ci) {
                        bool partial = false;
                        const uint dv = data[(k + ci) * w + i];
                        if (agg && ci == runlen) {
                            partial = true;
                        } else if (!(*f & ((T1_SIGMA_THIS | T1_PI_THIS) << (ci * 3U)))) {
                            const uint ctxt1 = lut_zc[(*f >> (ci * 3U)) & T1_SIGMA_NEIGHBOURS];
                            const uint v = (dv & one) ? 1u : 0u;
                            mq_encode(m, ctxs, ctxt1, v, out);
                            partial = v != 0;
                        }
                        if (partial) {
                            const uint lu = sc_index(*f, f[-1], f[1], ci);
                            nmsedec += nmsedec_sig(dv & 0x7fffffffu, (uint)bpno);
                            const uint v = dv >> 31;
                            mq_encode(m, ctxs, lut_ctxno_sc[lu], v ^ lut_spb[lu], out);
                            update_flags(f, ci, v, stride);
                        }
                        *f &= ~(T1_PI_THIS << (3U * ci));
                    }
                }
            }
        }

        if (passno < MAXPASSES) {
            pr[passno].nmsedec = nmsedec;
            if (passtype == 2 && bpno == 0) {
                mq_flush(m, out);
                pr[passno].rate = (uint)m.bp;
            } else {
                pr[passno].rate = (uint)m.bp + 3u;
            }
        }
        if (++passtype == 3) {
            passtype = 0;
            bpno--;
        }
    }
    if (passno > MAXPASSES) {
        res->status = 1;
        passno = MAXPASSES;
    }
    res->totalpasses = passno;
    finish_rates(pr, passno, (uint)m.bp, out);
    res->nbytes = pr[passno - 1].rate;
}

/* ======================================================================== */
/* Design (b), after Matela, Rusnak, Holub (DCC 2011), in two kernels of   */
/* the same command buffer:                                                 */
/*                                                                          */
/* t1_cm_simd: context modelling. One SIMD-group (32 lanes) per code-block */
/*   (w, h <= 32), lane x owns column x. Each column is kept as 32-bit      */
/*   row masks (bit y = row y): magnitude bit of the current bit-plane M,  */
/*   significance before the bit-plane S0 (= first significant bit-plane   */
/*   above b), "already refined" MU, sign SG. Pass membership and all      */
/*   contexts follow from shifts/ORs of the masks of the column and of its */
/*   two neighbour columns (simd_shuffle). The only causal dependency, the */
/*   significance propagation inside the sigpass, is solved as the least   */
/*   fixed point of a monotone mask equation: in-column propagation with a */
/*   Kogge-Stone prefix, across columns by iterating until no lane changes */
/*   (usually 2-4 iterations). The time rule "neighbour already visited in  */
/*   this pass or not" is resolved per neighbour direction (see A/B below). */
/*   The (context, decision) symbols of each bit-plane are generated per   */
/*   stripe, lanes placing their column's symbols with a SIMD prefix sum   */
/*   into threadgroup memory, and then copied to a device symbol pool.     */
/* t1_mq_serial: one thread per code-block runs the MQ coder over its      */
/*   symbol stream (state in registers, context states in threadgroup      */
/*   memory), terminates the last pass and does OpenJPEG's rate fix-ups.   */
/* ======================================================================== */

#define CM_SG_PER_TG 4
#define CM_STAGE_BYTES 2816 /* 2 symbols per sample + 3 per stripe column */

#define ROWS_R0 0x11111111u
#define ROWS_R3 0x88888888u

/* NS: rows of p reachable from a row of g through consecutive rows of p,  */
/* going down (y-1 -> y).                                                  */
static inline uint prop_down(uint g, uint p)
{
    uint P = p;
    g |= P & (g << 1);
    P &= P << 1;
    g |= P & (g << 2);
    P &= P << 2;
    g |= P & (g << 4);
    P &= P << 4;
    g |= P & (g << 8);
    P &= P << 8;
    g |= P & (g << 16);
    return g;
}

/* Neighbourhood of row y, "A" = state of the neighbours already visited  */
/* in the pass, "B" = state of those not yet visited. Visited before      */
/* (x, y): N, NW, W; NE if y is the first row of its stripe; SW unless y   */
/* is the last row of its stripe. Returns the 9-bit ZC index of t1.c      */
/* (bit0 NW, 1 N, 2 NE, 3 W, 5 E, 6 SW, 7 S, 8 SE).                        */
static inline uint zc_index(uint y, uint A, uint AL, uint AR, uint B, uint BL,
                            uint BR)
{
    const uint r = y & 3u;
    uint up = y ? (y - 1) : 32; /* 32 = out of range: shift gives 0 */
    uint idx = 0;
    if (y > 0) {
        idx |= ((AL >> up) & 1u) << 0;
        idx |= ((A >> up) & 1u) << 1;
        idx |= (((r == 0 ? AR : BR) >> up) & 1u) << 2;
    }
    idx |= ((AL >> y) & 1u) << 3;
    idx |= ((BR >> y) & 1u) << 5;
    if (y < 31) {
        idx |= (((r == 3 ? BL : AL) >> (y + 1)) & 1u) << 6;
        idx |= ((B >> (y + 1)) & 1u) << 7;
        idx |= ((BR >> (y + 1)) & 1u) << 8;
    }
    return idx;
}

/* Sign context / sign prediction index of t1.c (T1_LUT_* bits) */
static inline uint sc_lu(uint y, uint A, uint AL, uint B, uint BR,
                         uint SG, uint SGL, uint SGR)
{
    uint lu = 0;
    uint w = (AL >> y) & 1u;
    uint e = (BR >> y) & 1u;
    lu |= w << 3;
    lu |= (w & (SGL >> y)) << 0;
    lu |= e << 5;
    lu |= (e & (SGR >> y)) << 2;
    if (y > 0) {
        uint n = (A >> (y - 1)) & 1u;
        lu |= n << 1;
        lu |= (n & (SG >> (y - 1))) << 4;
    }
    if (y < 31) {
        uint s = (B >> (y + 1)) & 1u;
        lu |= s << 7;
        lu |= (s & (SG >> (y + 1))) << 6;
    }
    return lu;
}

#define SYM(ctx, d) ((uchar)((ctx) | ((d) << 7)))

/* Memory: the symbols of a code-block are stored in its own input region */
/* (data_off, d.region bytes, at least 256), which is free once the input  */
/* sits in threadgroup memory, from byte 16 on; bit-planes that do not fit */
/* go to the shared pool. The MQ kernel then writes the coded bytes into   */
/* the same region, behind the symbols it has not read yet. When the pool  */
/* or the pass records are full, the input is written back and the block   */
/* is marked ST_RETRY for another dispatch.                                */
kernel void t1_cm_simd(device const cblk_desc* descs [[buffer(0)]],
                       device int* data_buf [[buffer(1)]],
                       device uchar* pool [[buffer(2)]],
                       device atomic_uint* counters [[buffer(3)]],
                       device cblk_res* res_buf [[buffer(4)]],
                       device pass_res* pass_buf [[buffer(5)]],
                       constant uint& count [[buffer(6)]],
                       constant uint2& caps [[buffer(7)]], /* pool bytes, passes */
                       device const uint* idx [[buffer(8)]],
                       constant uint& use_idx [[buffer(9)]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint x [[thread_index_in_simdgroup]])
{
    threadgroup uint abs_tg[CM_SG_PER_TG][32 * 32];
    threadgroup uint4 stage_tg[CM_SG_PER_TG][CM_STAGE_BYTES / 16];

    const uint k0 = tgid * CM_SG_PER_TG + sg;
    if (k0 >= count) {
        return;
    }
    const uint cb = use_idx ? idx[k0] : k0;
    const cblk_desc d = descs[cb];
    const uint w = d.w_h & 0xffffu;
    const uint h = d.w_h >> 16;
    const uint orient = d.orient_sty & 0xffu;
    device int* data = data_buf + d.data_off;
    device uchar* region = (device uchar*)data;
    const uint R = d.region;
    device cblk_res* res = res_buf + cb;
    if (w > 32 || h > 32) {
        /* left to t1_enc_cblk_serial */
        if (x == 0) {
            res->status = ST_BIG;
        }
        return;
    }
    constant uchar* lut_zc = lut_ctxno_zc + (orient << 9);
    threadgroup uint* absv = abs_tg[sg];
    threadgroup uchar* stage = (threadgroup uchar*)stage_tg[sg];

    /* Load column x: magnitudes to threadgroup memory, sign mask */
    const bool active = x < w;
    /* rows of this lane's column that exist */
    const uint valid = !active ? 0u : (h >= 32) ? 0xffffffffu : ((1u << h) - 1u);
    uint SG = 0;
    int maxv = 0;
    for (uint y = 0; y < h; ++y) {
        int t = active ? data[y * w + x] : 0;
        uint a;
        if (t < 0) {
            if (t == INT_MIN) {
                t = INT_MIN + 1;
            }
            a = (uint)(-t);
            SG |= 1u << y;
        } else {
            a = (uint)t;
        }
        maxv = max(maxv, (int)a);
        absv[y * 32 + x] = a;
    }
    maxv = simd_max(maxv);
    const uint numbps = maxv ? (uint)((int)(31 - clz((uint)maxv)) + 1 -
                                      T1_NMSEDEC_FRACBITS) : 0u;
    if (x == 0) {
        res->numbps = numbps;
        res->status = 0;
        res->nbytes = 0;
    }
    if (numbps == 0 || (int)numbps - 1 < 0) {
        if (x == 0) {
            res->totalpasses = 0;
        }
        return;
    }
    /* pass records */
    const uint npasses = min(3u * numbps - 2u, (uint)MAXPASSES);
    uint poff = 0;
    if (x == 0) {
        poff = atomic_fetch_add_explicit(&counters[1], npasses,
                                         memory_order_relaxed);
    }
    poff = simd_broadcast(poff, 0);
    if (poff + npasses > caps.y) {
        if (x == 0) {
            res->status = ST_RETRY;
        }
        return;
    }
    device pass_res* pr = pass_buf + poff;
    uint rpos = 16; /* next free byte of the region */
    /* the whole column has been read before the region is written */
    simdgroup_barrier(mem_flags::mem_device);
    const uint SGL = (x == 0) ? 0u : simd_shuffle_up(SG, 1);
    const uint SGR = (x == 31) ? 0u : simd_shuffle_down(SG, 1);

    uint passno = 0;
    for (int bpno = (int)numbps - 1; bpno >= 0; --bpno) {
        const uint b = (uint)bpno;
        /* Masks of the bit-plane */
        uint M = 0, S0 = 0, MU = 0;
        for (uint y = 0; y < h; ++y) {
            const uint a = absv[y * 32 + x] >> (b + T1_NMSEDEC_FRACBITS);
            M |= (a & 1u) << y;
            S0 |= (a > 1u ? 1u : 0u) << y;
            MU |= (a > 3u ? 1u : 0u) << y;
        }
        const uint S0L = (x == 0) ? 0u : simd_shuffle_up(S0, 1);
        const uint S0R = (x == 31) ? 0u : simd_shuffle_down(S0, 1);
        const bool first = (bpno == (int)numbps - 1);
        uint base = 0;
        uint coded = 0;
        uint A = S0;       /* S0 | NS_sp */
        uint AL = S0L, AR = S0R;

        if (!first) {
            /* ---- significance propagation: least fixed point ---- */
            const uint cand = valid & ~S0;
            const uint extc = (S0 >> 1) | (S0 << 1) | S0R | (S0R >> 1);
            for (;;) {
                const uint ext = extc | AL | (AL << 1) |
                                 (((S0L & ROWS_R0) | (AL & ~ROWS_R0)) >> 1) |
                                 (((AR & ROWS_R3) | (S0R & ~ROWS_R3)) << 1);
                const uint p = cand & M;
                const uint NS = prop_down(p & ext, p);
                const uint An = S0 | NS;
                const bool done = simd_all(An == A);
                A = An;
                AL = (x == 0) ? 0u : simd_shuffle_up(A, 1);
                AR = (x == 31) ? 0u : simd_shuffle_down(A, 1);
                if (done) {
                    coded = cand & (ext | (A << 1));
                    break;
                }
            }
            /* symbols */
            int nmse = 0;
            for (uint s = 0; s < h; s += 4) {
                const uint sm = (0xfu << s) & valid;
                const uint cnt = popcount(coded & sm) + popcount(A & ~S0 & sm);
                const uint off = base + simd_prefix_exclusive_sum(cnt);
                uint o = off;
                uint todo = coded & sm;
                while (todo) {
                    const uint y = ctz(todo);
                    todo &= todo - 1;
                    const uint v = (M >> y) & 1u;
                    stage[o++] = SYM(lut_zc[zc_index(y, A, AL, AR, S0, S0L, S0R)], v);
                    if (v) {
                        const uint lu = sc_lu(y, A, AL, S0, S0R, SG, SGL, SGR);
                        stage[o++] = SYM(lut_ctxno_sc[lu], ((SG >> y) & 1u) ^ lut_spb[lu]);
                        nmse += nmsedec_sig(absv[y * 32 + x], b);
                    }
                }
                base = simd_broadcast(off + cnt, 31);
            }
            nmse = simd_sum(nmse);
            if (x == 0 && passno < npasses) {
                pr[passno].nmsedec = nmse;
                pr[passno].rate = base; /* symbol count */
            }
            passno++;

            /* ---- magnitude refinement ---- */
            const uint pbase = base;
            nmse = 0;
            const uint AN = (A << 1) | (A >> 1) | AL | (AL << 1) | (AL >> 1) |
                            AR | (AR << 1) | (AR >> 1);
            for (uint s = 0; s < h; s += 4) {
                const uint sm = (0xfu << s) & valid;
                const uint cnt = popcount(S0 & sm);
                const uint off = base + simd_prefix_exclusive_sum(cnt);
                uint o = off;
                uint todo = S0 & sm;
                while (todo) {
                    const uint y = ctz(todo);
                    todo &= todo - 1;
                    uint ctx = ((AN >> y) & 1u) ? CTX_MAG + 1 : CTX_MAG;
                    ctx = ((MU >> y) & 1u) ? CTX_MAG + 2 : ctx;
                    stage[o++] = SYM(ctx, (M >> y) & 1u);
                    nmse += nmsedec_ref(absv[y * 32 + x], b);
                }
                base = simd_broadcast(off + cnt, 31);
            }
            nmse = simd_sum(nmse);
            if (x == 0 && passno < npasses) {
                pr[passno].nmsedec = nmse;
                pr[passno].rate = base - pbase;
            }
            passno++;
        }

        /* ---- cleanup ---- */
        {
            const uint pbase = base;
            const uint cand = valid & ~A & ~coded;   /* not significant, not coded */
            const uint NSc = cand & M;
            const uint Ac = A | NSc;                 /* visited state */
            const uint AcL = (x == 0) ? 0u : simd_shuffle_up(Ac, 1);
            const uint AcR = (x == 31) ? 0u : simd_shuffle_down(Ac, 1);
            /* B state = A (S0 | NS_sp), neighbours AL / AR */
            int nmse = 0;
            for (uint s = 0; s < h; s += 4) {
                const uint sm = (0xfu << s) & valid;
                const uint own = cand & sm;
                bool agg = false;
                uint runlen = 0;
                if (active && s + 4 <= h && own == sm) {
                    /* neighbourhood of the stripe column as in the flags */
                    /* word: rows s-1..s+4 of columns x-1, x, x+1          */
                    const uint rows_above = (s > 0) ? (1u << (s - 1)) : 0u;
                    const uint rows_in = sm;
                    const uint row_below = (s + 4 < 32) ? (1u << (s + 4)) : 0u;
                    uint nb = (AcL & (rows_above | rows_in)) | (AL & row_below) |
                              (Ac & rows_above) | (A & row_below) |
                              (AcR & rows_above) | (AR & (rows_in | row_below));
                    agg = nb == 0;
                }
                uint cnt;
                uint todo;
                if (agg) {
                    const uint mr = (M >> s) & 0xfu;
                    runlen = mr ? ctz(mr) : 4u;
                    if (runlen == 4) {
                        cnt = 1;
                        todo = 0;
                    } else {
                        todo = own & ~((2u << (s + runlen)) - 1u);
                        cnt = 4 + popcount(todo) + popcount(todo & M);
                    }
                } else {
                    todo = own;
                    cnt = popcount(todo) + popcount(todo & M);
                }
                const uint off = base + simd_prefix_exclusive_sum(cnt);
                uint o = off;
                if (agg) {
                    stage[o++] = SYM(CTX_AGG, runlen != 4 ? 1u : 0u);
                    if (runlen != 4) {
                        const uint y = s + runlen;
                        stage[o++] = SYM(CTX_UNI, runlen >> 1);
                        stage[o++] = SYM(CTX_UNI, runlen & 1u);
                        const uint lu = sc_lu(y, Ac, AcL, A, AR, SG, SGL, SGR);
                        stage[o++] = SYM(lut_ctxno_sc[lu], ((SG >> y) & 1u) ^ lut_spb[lu]);
                        nmse += nmsedec_sig(absv[y * 32 + x], b);
                    }
                }
                while (todo) {
                    const uint y = ctz(todo);
                    todo &= todo - 1;
                    const uint v = (M >> y) & 1u;
                    stage[o++] = SYM(lut_zc[zc_index(y, Ac, AcL, AcR, A, AL, AR)], v);
                    if (v) {
                        const uint lu = sc_lu(y, Ac, AcL, A, AR, SG, SGL, SGR);
                        stage[o++] = SYM(lut_ctxno_sc[lu], ((SG >> y) & 1u) ^ lut_spb[lu]);
                        nmse += nmsedec_sig(absv[y * 32 + x], b);
                    }
                }
                base = simd_broadcast(off + cnt, 31);
            }
            nmse = simd_sum(nmse);
            if (x == 0 && passno < npasses) {
                pr[passno].nmsedec = nmse;
                pr[passno].rate = base - pbase;
            }
            passno++;
        }

        /* ---- store the bit-plane's symbols: region, else pool ---- */
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint T = (base + 15u) & ~15u;
        uint loc;
        device uint4* dst;
        if (rpos + T <= R) {
            loc = rpos;
            dst = (device uint4*)(region + rpos);
            rpos += T;
        } else {
            uint chunk = 0;
            if (x == 0) {
                chunk = atomic_fetch_add_explicit(&counters[0], T,
                                                  memory_order_relaxed);
            }
            chunk = simd_broadcast(chunk, 0);
            if (chunk + T > caps.x) {
                /* pool full: give the input back for a retry */
                simdgroup_barrier(mem_flags::mem_device);
                if (active) {
                    for (uint y = 0; y < h; ++y) {
                        const int a = (int)absv[y * 32 + x];
                        data[y * w + x] = ((SG >> y) & 1u) ? -a : a;
                    }
                }
                if (x == 0) {
                    res->status = ST_RETRY;
                }
                return;
            }
            loc = chunk | LOC_POOL;
            dst = (device uint4*)(pool + chunk);
        }
        {
            threadgroup const uint4* src = (threadgroup const uint4*)stage;
            for (uint i = x; i < T / 16u; i += 32) {
                dst[i] = src[i];
            }
        }
        if (x == 0) {
            /* pass locations: the passes of the bit-plane are contiguous */
            const uint np = first ? 1u : 3u;
            uint o = loc;
            for (uint k = passno - np; k < passno; ++k) {
                if (k < npasses) {
                    pr[k].len = o;
                    o += pr[k].rate;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (x == 0) {
        res->totalpasses = min(passno, (uint)MAXPASSES);
        res->pass_off = poff;
        res->region_end = rpos;
        if (passno > MAXPASSES) {
            res->status = ST_PASSES;
        }
    }
}

#define MQ_TG 64

kernel void t1_mq_serial(device int* data_buf [[buffer(0)]],
                         device const uchar* pool [[buffer(1)]],
                         device const cblk_desc* descs [[buffer(2)]],
                         device cblk_res* res_buf [[buffer(4)]],
                         device pass_res* pass_buf [[buffer(5)]],
                         constant uint& count [[buffer(6)]],
                         device const uint* idx [[buffer(8)]],
                         constant uint& use_idx [[buffer(9)]],
                         uint tid [[thread_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uchar ctx_tg[MQ_TG][20];
    if (tid >= count) {
        return;
    }
    const uint gid = use_idx ? idx[tid] : tid;
    device cblk_res* res = res_buf + gid;
    const uint npasses = res->totalpasses;
    if (res->status != ST_OK || npasses == 0) {
        return;
    }
    const cblk_desc d = descs[gid];
    device uchar* region = (device uchar*)(data_buf + d.data_off);
    const uint R = d.region;
    const uint region_end = res->region_end;
    device pass_res* pr = pass_buf + res->pass_off;
    threadgroup uchar* ctxs = ctx_tg[lid];
    region_out ro;
    ro.p = region + 1;
    ro.lim = region_end > 16 ? 16 : R;
    ro.ovf = false;
    thread region_out* out = &ro;

    mq_enc m;
    mq_init(m, ctxs);
    region[0] = 0; /* fake byte */
    for (uint p = 0; p < npasses; ++p) {
        const uint n = pr[p].rate;
        const uint loc = pr[p].len;
        if (loc & LOC_POOL) {
            device const uchar* sp = pool + (loc & ~LOC_POOL);
            for (uint i = 0; i < n; ++i) {
                const uint s = sp[i];
                mq_encode(m, ctxs, s & 0x1fu, s >> 7, out);
            }
        } else {
            device const uchar* sp = region + loc;
            for (uint i = 0; i < n; ++i) {
                const uint s = sp[i];
                /* symbol loc + i is consumed: the region up to it is free */
                ro.lim = (loc + i + 1 >= region_end) ? R : loc + i + 1;
                mq_encode(m, ctxs, s & 0x1fu, s >> 7, out);
            }
        }
        if (p == npasses - 1) {
            mq_flush(m, out);
            pr[p].rate = (uint)m.bp;
        } else {
            pr[p].rate = (uint)m.bp + 3u;
        }
    }
    if (ro.ovf) {
        res->status = ST_OUTFULL;
        return;
    }
    finish_rates(pr, npasses, (uint)m.bp, out);
    res->nbytes = pr[npasses - 1].rate;
}
