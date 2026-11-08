# Copyright (c) 2026, CtrlMovie AG
# Copyright (c) 2026, Mart Jansink
# UNIT TESTS

include_directories(
  ${OPENJPEG_BINARY_DIR}/src/lib/openjp2 # opj_config.h
  ${OPENJPEG_SOURCE_DIR}/src/lib/openjp2
)

set(unit_test
  testempty0
  testempty1
  testempty2
)
foreach(ut ${unit_test})
  add_executable(${ut} ${ut}.c)
  target_link_libraries(${ut} openjp2)
  add_test(NAME ${ut} COMMAND ${ut})
endforeach()

# Fork extension: opj_decoder_set_extra_options() option checking
add_executable(testdecoderoptions testdecoderoptions.c)
target_link_libraries(testdecoderoptions openjp2)
add_test(NAME testdecoderoptions COMMAND testdecoderoptions)

# Fork extension: NEON DC level shift (dc_shift.h) against the scalar code
add_executable(testdcshift testdcshift.c)
target_link_libraries(testdcshift openjp2)
add_test(NAME testdcshift COMMAND testdcshift)

add_executable(testjp2 testjp2.c)
target_link_libraries(testjp2 openjp2)
add_test(NAME testjp2 COMMAND testjp2 ${OPJ_DATA_ROOT})

# CtrlMovie fork: zero-copy input (opj_image_set_external_data) and
# INPUT_MCT_DOMAIN=YES must produce byte-identical codestreams to the normal
# path. bench_encode is built with BUILD_UNIT_TESTS (src/lib/openjp2).
if(BUILD_UNIT_TESTS AND UNIX)
  add_test(NAME bench_encode_verify_cinema_mct COMMAND bench_encode -verify-modes -mode cinema -mct 1 -size 1024 540 -iters 2)
  add_test(NAME bench_encode_verify_cinema_nomct COMMAND bench_encode -verify-modes -mode cinema -mct 0 -size 1024 540 -iters 1)
  add_test(NAME bench_encode_verify_irr_mct COMMAND bench_encode -verify-modes -mode irr -mct 1 -size 1023 541 -iters 1)
  add_test(NAME bench_encode_verify_rev_mct COMMAND bench_encode -verify-modes -mode rev -mct 1 -size 1024 540 -iters 2)
  add_test(NAME bench_encode_verify_rev_nomct COMMAND bench_encode -verify-modes -mode rev -mct 0 -size 1024 540 -iters 1)
  # Cinema size limits measured from the tile-parts (SMPTE 429-2): the
  # synthetic frames are bound by the codestream limit, the grey noise frames
  # (almost all bytes in component 0 with -mct 1) by the per-component limit,
  # which they exceeded by the tile-part header bytes before the fix.
  add_test(NAME bench_encode_limits_cinema4k COMMAND bench_encode -check-limits -mode cinema -mct 1 -size 4096 2160 -iters 1)
  add_test(NAME bench_encode_limits_cinema2k COMMAND bench_encode -check-limits -mode cinema -mct 1 -size 2048 1080 -iters 1)
  add_test(NAME bench_encode_limits_cinema4k_comp COMMAND bench_encode -check-limits -mode cinema -mct 1 -size 4096 2160 -iters 1 -gray-noise 2 10)
  add_test(NAME bench_encode_limits_cinema2k_comp COMMAND bench_encode -check-limits -mode cinema -mct 1 -size 2048 1080 -iters 1 -gray-noise 5 10)
  # Exact codestream budget: rate-bound synthetic frames at the application's
  # bit rates (max_cs_size = rate / 8 / 24, max_comp_size = max_cs_size / 1.25)
  foreach(opj_br 50 75 125 250)
    add_test(NAME bench_encode_limits_2k_${opj_br}mbit COMMAND bench_encode -check-limits -bitrate ${opj_br} -mode cinema -mct 1 -size 2048 1080 -iters 1)
    add_test(NAME bench_encode_limits_4k_${opj_br}mbit COMMAND bench_encode -check-limits -bitrate ${opj_br} -mode cinema -mct 1 -size 4096 2160 -iters 1)
  endforeach()
endif()

# CtrlMovie fork: GPU tier-1 (METAL_T1=YES) must produce the CPU codestream.
if(BUILD_UNIT_TESTS AND UNIX AND APPLE AND OPJ_USE_METAL)
  add_test(NAME bench_encode_verify_metal_cinema_mct COMMAND bench_encode -verify-modes -metal -mode cinema -mct 1 -size 1024 540 -iters 2)
  add_test(NAME bench_encode_verify_metal_cinema_nomct COMMAND bench_encode -verify-modes -metal -mode cinema -mct 0 -size 1024 540 -iters 1)
  add_test(NAME bench_encode_verify_metal_irr_mct COMMAND bench_encode -verify-modes -metal -mode irr -mct 1 -size 1023 541 -iters 1)
  add_test(NAME bench_encode_verify_metal_rev_mct COMMAND bench_encode -verify-modes -metal -mode rev -mct 1 -size 1024 540 -iters 2)
  add_test(NAME bench_encode_verify_metal_rev_nomct COMMAND bench_encode -verify-modes -metal -mode rev -mct 0 -size 1024 540 -iters 1)
  add_test(NAME bench_encode_verify_metal_cinema4k COMMAND bench_encode -verify-modes -metal -mode cinema -mct 1 -size 4096 2160 -iters 1)
  # every 7th code-block through the CPU jobs (mixed GPU/CPU tile)
  add_test(NAME bench_encode_verify_metal_mixed COMMAND bench_encode -verify-modes -metal -mode irr -mct 1 -size 1023 541 -iters 1 -threads 4)
  set_tests_properties(bench_encode_verify_metal_mixed PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEBUG_CPU_EVERY=7")
  # simulated GPU failure: whole tile falls back to the CPU
  add_test(NAME bench_encode_verify_metal_gpufail COMMAND bench_encode -verify-modes -mode cinema -mct 1 -size 1024 540 -iters 1)
  set_tests_properties(bench_encode_verify_metal_gpufail PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEBUG_FAIL=1")
  # tiny symbol pool: code-blocks restored and re-run on the GPU
  add_test(NAME bench_encode_verify_metal_retry COMMAND bench_encode -verify-modes -metal -mode cinema -mct 1 -size 1024 540 -iters 1)
  set_tests_properties(bench_encode_verify_metal_retry PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEBUG_POOL=64")
  # every 5th code-block handed back to the CPU after the GPU run
  add_test(NAME bench_encode_verify_metal_refuse COMMAND bench_encode -verify-modes -metal -mode rev -mct 1 -size 1024 540 -iters 1)
  set_tests_properties(bench_encode_verify_metal_refuse PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEBUG_REFUSE_EVERY=5")
  add_test(NAME bench_encode_limits_metal_cinema4k_comp COMMAND bench_encode -check-limits -metal -mode cinema -mct 1 -size 4096 2160 -iters 1 -gray-noise 2 10)
  foreach(opj_br 50 75 125 250)
    add_test(NAME bench_encode_limits_metal_2k_${opj_br}mbit COMMAND bench_encode -check-limits -metal -bitrate ${opj_br} -mode cinema -mct 1 -size 2048 1080 -iters 1)
    add_test(NAME bench_encode_limits_metal_4k_${opj_br}mbit COMMAND bench_encode -check-limits -metal -bitrate ${opj_br} -mode cinema -mct 1 -size 4096 2160 -iters 1)
  endforeach()
  # 6 encoder instances sharing the process-wide Metal context
  add_test(NAME bench_encode_metal_concurrent COMMAND bench_encode -concurrent 6 -metal -mode cinema -mct 1 -size 2048 1080 -iters 3)
endif()

# Fork extension: the decoder. Every decoding of bench_decode -verify (whole
# image and a window, reduce 0..2, 1 and 4 threads) must give the samples of
# the CPU decoder; with -output-mct-domain the planes must also give them
# back through the documented inverse MCT and DC shift.
if(BUILD_UNIT_TESTS AND UNIX)
  add_test(NAME bench_decode_verify_mctdomain_irr COMMAND bench_decode -verify -output-mct-domain -mode cinema -mct 1 -size 1024 540)
  add_test(NAME bench_decode_verify_mctdomain_rev COMMAND bench_decode -verify -output-mct-domain -mode rev -mct 1 -size 1000 600)
endif()

# Fork extension: GPU tier-1 decoding (METAL_T1=YES) must give the samples of
# the CPU decoder.
if(BUILD_UNIT_TESTS AND UNIX AND APPLE AND OPJ_USE_METAL)
  add_test(NAME bench_decode_verify_metal_cinema_mct COMMAND bench_decode -verify -require-gpu -mode cinema -mct 1 -size 1024 540)
  add_test(NAME bench_decode_verify_metal_cinema_nomct COMMAND bench_decode -verify -require-gpu -mode cinema -mct 0 -size 1024 540)
  add_test(NAME bench_decode_verify_metal_irr_mct COMMAND bench_decode -verify -require-gpu -mode irr -mct 1 -size 1023 541)
  add_test(NAME bench_decode_verify_metal_rev_mct COMMAND bench_decode -verify -require-gpu -mode rev -mct 1 -size 1000 600)
  add_test(NAME bench_decode_verify_metal_rev_nomct COMMAND bench_decode -verify -require-gpu -mode rev -mct 0 -size 777 333)
  add_test(NAME bench_decode_verify_metal_cinema4k COMMAND bench_decode -verify -require-gpu -mode cinema -mct 1 -size 4096 2160 -threads 8)
  # 64x64 code-blocks, several quality layers (several chunks per code-block)
  add_test(NAME bench_decode_verify_metal_cblk64 COMMAND bench_decode -verify -require-gpu -mode irr -mct 1 -size 1000 600 -cblk 64)
  add_test(NAME bench_decode_verify_metal_layers COMMAND bench_decode -verify -require-gpu -mode irr -mct 1 -size 1000 600 -layers 5)
  # BYPASS|RESET: all code-blocks go to the CPU
  add_test(NAME bench_decode_verify_metal_cblksty COMMAND bench_decode -verify -mode irr -mct 1 -size 600 400 -cblksty 3)
  add_test(NAME bench_decode_verify_metal_mctdomain COMMAND bench_decode -verify -require-gpu -output-mct-domain -mode cinema -mct 1 -size 1024 540)
  # every 7th code-block on the CPU
  add_test(NAME bench_decode_verify_metal_mixed COMMAND bench_decode -verify -require-gpu -mode cinema -mct 1 -size 1024 540)
  set_tests_properties(bench_decode_verify_metal_mixed PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_DEBUG_CPU_EVERY=7")
  # GPU runs reported as failed after writing: all redone on the CPU
  add_test(NAME bench_decode_verify_metal_gpufail COMMAND bench_decode -verify -mode cinema -mct 1 -size 1024 540)
  set_tests_properties(bench_decode_verify_metal_gpufail PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_DEBUG_FAIL=1")
  # every 5th GPU code-block handed back to the CPU
  add_test(NAME bench_decode_verify_metal_refuse COMMAND bench_decode -verify -require-gpu -mode rev -mct 1 -size 1000 600)
  set_tests_properties(bench_decode_verify_metal_refuse PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_DEBUG_REFUSE_EVERY=5;OPJ_METAL_T1_DEC_POISON=1")
  add_test(NAME bench_decode_verify_metal_nozerocopy COMMAND bench_decode -verify -require-gpu -mode cinema -mct 1 -size 1024 540)
  set_tests_properties(bench_decode_verify_metal_nozerocopy PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_ZEROCOPY=0")
  # all on the GPU, and nearly all on the CPU
  add_test(NAME bench_decode_verify_metal_allgpu COMMAND bench_decode -verify -require-gpu -mode irr -mct 1 -size 1000 600)
  set_tests_properties(bench_decode_verify_metal_allgpu PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_CPU_SHARE=0")
  add_test(NAME bench_decode_verify_metal_mostcpu COMMAND bench_decode -verify -require-gpu -mode rev -mct 1 -size 1000 600 -threads 8)
  set_tests_properties(bench_decode_verify_metal_mostcpu PROPERTIES ENVIRONMENT "OPJ_METAL_T1_DEC_CPU_SHARE=95")
  # 6 decoders sharing the process-wide Metal context
  add_test(NAME bench_decode_metal_concurrent COMMAND bench_decode -concurrent 6 -metal -mode cinema -mct 1 -size 2048 1080 -iters 3)
endif()
