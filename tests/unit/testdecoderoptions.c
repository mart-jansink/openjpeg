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


/* Fork extension: opj_decoder_set_extra_options() accepts the documented
 * options and values and rejects everything else. */

#include <stdio.h>
#include "openjpeg.h"

static int g_failures = 0;

static void quiet(const char* msg, void* client_data)
{
    (void)msg;
    (void)client_data;
}

static void check(OPJ_CODEC_FORMAT fmt, const char* opt, OPJ_BOOL expected)
{
    opj_dparameters_t params;
    const char* options[2];
    opj_codec_t* codec = opj_create_decompress(fmt);
    OPJ_BOOL got;
    opj_set_default_decoder_parameters(&params);
    opj_set_error_handler(codec, quiet, NULL);
    opj_set_warning_handler(codec, quiet, NULL);
    opj_setup_decoder(codec, &params);
    options[0] = opt;
    options[1] = NULL;
    got = opj_decoder_set_extra_options(codec, opt ? options : NULL);
    if (got != expected) {
        fprintf(stderr, "%s %s: got %d, expected %d\n",
                fmt == OPJ_CODEC_J2K ? "J2K" : "JP2", opt ? opt : "(null)",
                got, expected);
        g_failures++;
    }
    opj_destroy_codec(codec);
}

int main(int argc, char **argv)
{
    const char* options[] = { "METAL_T1=YES", NULL };
    opj_codec_t* enc;
    (void)argc;
    (void)argv;

    check(OPJ_CODEC_J2K, NULL, OPJ_TRUE);
    check(OPJ_CODEC_J2K, "METAL_T1=YES", OPJ_TRUE);
    check(OPJ_CODEC_J2K, "METAL_T1=NO", OPJ_TRUE);
    check(OPJ_CODEC_J2K, "METAL_T1=MAYBE", OPJ_FALSE);
    check(OPJ_CODEC_J2K, "OUTPUT_MCT_DOMAIN=YES", OPJ_TRUE);
    check(OPJ_CODEC_J2K, "OUTPUT_MCT_DOMAIN=NO", OPJ_TRUE);
    check(OPJ_CODEC_J2K, "OUTPUT_MCT_DOMAIN=1", OPJ_FALSE);
    check(OPJ_CODEC_J2K, "NO_SUCH_OPTION=YES", OPJ_FALSE);
    check(OPJ_CODEC_JP2, "METAL_T1=YES", OPJ_TRUE);
    check(OPJ_CODEC_JP2, "OUTPUT_MCT_DOMAIN=NO", OPJ_TRUE);
    /* JP2 colour, palette and channel definitions assume samples */
    check(OPJ_CODEC_JP2, "OUTPUT_MCT_DOMAIN=YES", OPJ_FALSE);

    /* decoder options on an encoder */
    enc = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(enc, quiet, NULL);
    if (opj_decoder_set_extra_options(enc, options)) {
        fprintf(stderr, "decoder options accepted by an encoder\n");
        g_failures++;
    }
    opj_destroy_codec(enc);

    if (g_failures == 0) {
        printf("PASS\n");
    }
    return g_failures ? 1 : 0;
}
