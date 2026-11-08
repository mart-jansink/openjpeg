# Copyright (c) 2026, CtrlMovie AG
# Copyright (c) 2026, Mart Jansink
# Metal T1 encoder and decoder (OPJ_USE_METAL), included from
# src/lib/openjp2/CMakeLists.txt so that it runs in that directory scope.
set(OPJ_METAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/metal)
set(OPJ_METAL_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/metal)
file(MAKE_DIRECTORY ${OPJ_METAL_GEN_DIR})
add_custom_command(
  OUTPUT ${OPJ_METAL_GEN_DIR}/opj_metal_t1_src.h
  COMMAND ${CMAKE_COMMAND}
    -DMETAL_SRC=${OPJ_METAL_DIR}/t1_enc.metal
    -DLUTS=${CMAKE_CURRENT_SOURCE_DIR}/t1_luts.h
    -DOUT_METAL=${OPJ_METAL_GEN_DIR}/t1_enc_full.metal
    -DOUT_HEADER=${OPJ_METAL_GEN_DIR}/opj_metal_t1_src.h
    -P ${OPJ_METAL_DIR}/opj_metal_embed.cmake
  DEPENDS ${OPJ_METAL_DIR}/t1_enc.metal ${CMAKE_CURRENT_SOURCE_DIR}/t1_luts.h
          ${OPJ_METAL_DIR}/opj_metal_embed.cmake
  COMMENT "Embedding Metal T1 shader source")
add_custom_command(
  OUTPUT ${OPJ_METAL_GEN_DIR}/opj_metal_t1_dec_src.h
  COMMAND ${CMAKE_COMMAND}
    -DMETAL_SRC=${OPJ_METAL_DIR}/t1_dec.metal
    -DLUTS=${CMAKE_CURRENT_SOURCE_DIR}/t1_luts.h
    -DOUT_METAL=${OPJ_METAL_GEN_DIR}/t1_dec_full.metal
    -DOUT_HEADER=${OPJ_METAL_GEN_DIR}/opj_metal_t1_dec_src.h
    -DSYMBOL=opj_metal_t1_dec_src
    -P ${OPJ_METAL_DIR}/opj_metal_embed.cmake
  DEPENDS ${OPJ_METAL_DIR}/t1_dec.metal ${CMAKE_CURRENT_SOURCE_DIR}/t1_luts.h
          ${OPJ_METAL_DIR}/opj_metal_embed.cmake
  COMMENT "Embedding Metal T1 decoder shader source")
set_source_files_properties(${OPJ_METAL_DIR}/opj_metal_common.m PROPERTIES
  COMPILE_OPTIONS "-fobjc-arc")
set_source_files_properties(${OPJ_METAL_DIR}/opj_metal_t1_dec.m PROPERTIES
  COMPILE_OPTIONS "-fobjc-arc"
  OBJECT_DEPENDS ${OPJ_METAL_GEN_DIR}/opj_metal_t1_dec_src.h)
set_source_files_properties(${OPJ_METAL_DIR}/opj_metal_t1.m PROPERTIES
  COMPILE_OPTIONS "-fobjc-arc"
  OBJECT_DEPENDS ${OPJ_METAL_GEN_DIR}/opj_metal_t1_src.h)
target_sources(${OPENJPEG_LIBRARY_NAME} PRIVATE
  ${OPJ_METAL_DIR}/opj_metal_t1.m
  ${OPJ_METAL_DIR}/opj_metal_t1.h
  ${OPJ_METAL_GEN_DIR}/opj_metal_t1_src.h
  ${OPJ_METAL_DIR}/opj_metal_common.m
  ${OPJ_METAL_DIR}/opj_metal_common.h
  ${OPJ_METAL_DIR}/opj_metal_t1_dec.m
  ${OPJ_METAL_DIR}/opj_metal_t1_dec.h
  ${OPJ_METAL_GEN_DIR}/opj_metal_t1_dec_src.h)
target_include_directories(${OPENJPEG_LIBRARY_NAME} PRIVATE
  ${OPJ_METAL_GEN_DIR} ${OPJ_METAL_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${OPENJPEG_LIBRARY_NAME}
  "-framework Metal" "-framework Foundation")
if(BUILD_UNIT_TESTS)
  add_executable(test_metal_t1 ${OPJ_METAL_DIR}/test_metal_t1.c)
  target_include_directories(test_metal_t1 PRIVATE ${OPJ_METAL_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_metal_t1 ${OPENJPEG_LIBRARY_NAME}
    "-framework Metal" "-framework Foundation")
endif()
target_compile_definitions(${OPENJPEG_LIBRARY_NAME} PRIVATE OPJ_USE_METAL=1)
foreach(opj_bench bench_encode bench_decode)
  if(TARGET ${opj_bench})
    target_compile_definitions(${opj_bench} PRIVATE OPJ_USE_METAL=1)
    target_include_directories(${opj_bench} PRIVATE ${OPJ_METAL_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endforeach()
