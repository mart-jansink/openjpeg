# OPENJPEG Library and Applications

## CtrlMovie fork: Metal-accelerated encoder for Apple Silicon

This repository is CtrlMovie AG's fork of OpenJPEG 2.5.4, branch `metal-encoder`.
It adds a GPU (Metal) path to the JPEG 2000 **encoder** for Cinema 2K/4K DCP export
on Apple Silicon, plus CPU improvements that keep the codestream byte-identical to
upstream OpenJPEG. A Metal decoder was added later; see "Metal Tier-1 decoder" below.
Copyright (c) 2026, CtrlMovie AG on the changed and added files; the BSD 2-clause
license is unchanged.

### What is added

| Area | Addition |
| --- | --- |
| GPU Tier-1 encoder | `src/lib/openjp2/metal/`: EBCOT context modelling and MQ coding in Metal, one SIMD group per code-block, byte-identical to the CPU encoder. Enabled per encoder with the extra option `METAL_T1=YES` (or env `OPJ_METAL_T1=1`). Code-blocks the GPU cannot take (code-block style other than 0) fall back to the CPU path automatically. |
| Zero-copy input | `opj_image_set_external_data()`: the encoder uses caller-owned planes (for example a shared `MTLBuffer`) in place and never frees them. Feature macro `OPJ_HAVE_EXTERNAL_INPUT_DATA`. |
| Pre-transformed input | Extra option `INPUT_MCT_DOMAIN=YES`: the caller has already applied the DC level shift and the forward MCT (float planes for 9/7), so the encoder skips both stages. Single-tile encodes only; see the contract in `openjpeg.h`. |
| CPU speedups | NEON forward 9/7 DWT; rate-allocation caches in `opj_tcd_makelayer` and the trial T2 passes. Bit-identical output. |
| Cinema compliance | The per-component size limit (`max_comp_size`) counts tile-part headers, as SMPTE 429-2 measures it, and the codestream byte budget for `max_cs_size` is computed exactly (64-bit, rounding down, real header sizes), so rate-bound frames never exceed the limit. Generic ratio-based encodes are unchanged. |
| Tools and tests | `bench_encode` (in-memory encode benchmark with `-verify-modes`, `-metal`, `-concurrent`, `-check-limits`), `test_metal_t1` (GPU vs CPU code-block oracle on `OPJ_T1_DUMP` dumps), CTests `bench_encode_verify_*` and `bench_encode_limits_*` (Cinema size limits at 50 to 250 Mbit/s). `bench_encode -bitrate MBIT` mirrors the app's rate settings. `OPJ_ENC_PROFILE=1` prints per-stage timings. |

### Building on macOS

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DOPJ_USE_METAL=ON -DBUILD_UNIT_TESTS=ON -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -R "bench_encode|tte"
```

`OPJ_USE_METAL` (default OFF) compiles the Objective-C/Metal code into `libopenjp2`
and defines `OPJ_HAVE_METAL_T1` in `opj_config.h`; the shader is compiled at run time
from embedded source. Consumers of the arm64 slice must link `Metal.framework` and
`Foundation.framework`. A universal library can be made by building an arm64 slice
with `OPJ_USE_METAL=ON` and an x86_64 slice with it OFF and merging them with `lipo`.

### Measured on an M4 Max (Cinema 4K, 3840x2076, 12-bit, 9/7, 250 Mbit/s)

| | CPU path (upstream) | Metal path |
| --- | --- | --- |
| One 4K frame, single-threaded encoder | 1270 ms | about 95 ms |
| 12 concurrent encoders | 8.5 fps | about 60 fps |

The memory footprint of the GPU path is about 180 MB per in-flight batch, pool of 3
by default (`OPJ_METAL_T1_MAX_BATCHES`).

### Notes

* Upstream OpenJPEG is unmaintained (see below); this fork is maintained for
  CtrlMovie's DCP export and is not intended for upstream submission.
* The `reference_frames/` directory (test images) is not part of the repository.


## Metal Tier-1 decoder

Added on top of the encoder work above by Mart Jansink (Copyright (c) 2026, Mart
Jansink on the changed and added files; same BSD 2-clause license). Decoded samples
are bit-identical to the CPU decoder.

| Area | Addition |
| --- | --- |
| Decoder extra options | `opj_decoder_set_extra_options()` (feature macro `OPJ_HAVE_DECODER_EXTRA_OPTIONS`), the decoder counterpart of `opj_encoder_set_extra_options()`. Unknown options are an error; see the list in `openjpeg.h`. |
| GPU Tier-1 decoder | `src/lib/openjp2/metal/t1_dec.metal` and `opj_metal_t1_dec.{h,m}`: MQ decoding and the significance, refinement and cleanup passes, one GPU thread per code-block. Enabled per decoder with `METAL_T1=YES` (or env `OPJ_METAL_T1_DECODE=1`; separate from the encoder's `OPJ_METAL_T1`). Code-blocks the GPU cannot take (code-block style other than 0, ROI shift, more than one segment, larger than 64x64) are decoded on the CPU; if the GPU fails, the whole tile is redone on the CPU. |
| Adaptive CPU/GPU split | Decoding is causal, so a GPU thread takes far longer than a CPU core on a long code-block (about 10 ms for a 1.6 KB, 36-pass block). The longest code-blocks, a share of the coded bytes, are decoded on the CPU while the GPU runs the rest. The share is chosen by measurement (`opj_t1_dec_split_choose()` in `t1.c`): per thread count and tile size, the T1 wall time per coded megabyte of the shares 0 to 95% and of "CPU only" is kept as an EWMA; each is measured once (outward from the best so far; a trial ends early when a decode takes 1.5 times the best one's cost, or 2 times for a warm-up decode, and candidates beyond one that lost by 1.5 times are skipped), then the fastest is used and its neighbours, the runner-up and CPU-only are re-probed now and then (runs of 4 decodes, the first 2 after any switch not counted, exponential back-off). `OPJ_METAL_T1_DEC_CPU_SHARE` (0 to 100) fixes the share; `OPJ_METAL_T1_DEC_ADAPT=0` uses the old formula (10% plus 5% per extra thread). |
| GPU admission | Never waits for the GPU: a decoder that finds the in-flight limit reached decodes the tile on the CPU. Each batch has its own command queue, so parallel decoders' batches overlap; the limit is 3 command buffers for decoders without a thread pool and 1 for decoders with one. `OPJ_METAL_T1_DEC_INFLIGHT` overrides it. |
| CPU: DC level shift | `dc_shift.h`: NEON row kernels for the decoder's DC level shift (bit-identical to the scalar code, checked by `testdcshift`), with the rows split over the decoder's thread pool. Speeds up the CPU and Metal paths alike; `OPJ_DEC_SCALAR_DC_SHIFT=1` restores the scalar code. |
| CPU: T1 job batching | Code-blocks go to the thread pool in batches of up to 64 code-blocks or 2 KB of coded data (upstream: one job per code-block, with three allocations and two trips through the pool's lock each). For the CPU decoder and the Metal decoder's CPU share; `OPJ_T1_DEC_BATCH=0` restores one job per code-block. |
| CPU: inverse 9/7 DWT | The vertical pass (`opj_v8dwt_decode_v_strips()` in `dwt.c`) does 4 strips of 8 columns (whole 128-byte cache lines) per sweep, pipelines the lifting down the column in chunks of 32 rows so that rows are written back soon after being read, prefetches the rows it is about to read, and uses one job per decoder thread (upstream: half). Its strip buffers are cached per worker thread. Fixes the 1.7 to 2× slowdown at widths of 2048 and 4096 (scope and full-container DCPs) and speeds up other widths too; bit-identical. `bench_dwt` gained `-width`, `-height`, `-iter` and `-hash`. |
| Zero-copy output | Tile planes are allocated page-aligned and wrapped with `newBufferWithBytesNoCopy`, so the GPU writes decoded code-blocks straight into the tile. Partial (area) decodes and planes that cannot be wrapped use a copy path. `OPJ_METAL_T1_DEC_ZEROCOPY=0` forces the copy path. |
| MCT-domain output | Extra option `OUTPUT_MCT_DOMAIN=YES` (J2K codecs only): the decoder skips the inverse MCT and DC level shift and returns the inverse-DWT output (int32 for 5/3, float bit patterns for 9/7), for callers that do colour conversion themselves. See the contract in `openjpeg.h`. |
| Tools and tests | `bench_decode` (`-verify` compares CPU and Metal decodes plane by plane over reduce levels, thread counts and a decode area, against a reference with the scalar DC shift; `-concurrent` with warm-up and a common start, `-warmup N`, `-output-mct-domain`, `-i file.j2c ...`), CTests `bench_decode_verify_*`, `testdecoderoptions` and `testdcshift`. `OPJ_DEC_PROFILE=1` prints per-stage timings; `OPJ_METAL_T1_DEC_PROFILE=1` prints per-batch GPU timings. Debug hooks: `OPJ_METAL_T1_DEC_DEBUG_CPU_EVERY`, `_DEBUG_FAIL`, `_DEBUG_REFUSE_EVERY`, `_POISON`. |

Build as above, then `ctest --test-dir build -R "bench_decode|testdecoderoptions|testdcshift"`.
`OPJ_HAVE_METAL_T1_DECODE` in `opj_config.h` says the decoder backend is compiled in.

### Measured on an M1 Max

The stock decoder (commit `67066820`, the base of this branch) against this branch's
CPU and Metal decoders, built the same way (Release, `OPJ_USE_METAL=ON`) and timed
by one harness through the public API only, creating a codec per decode as libdcp
does. Median ms per frame after 40 warm-up decodes per file (so that the adaptive
split has settled), the better of 2 interleaved rounds. 12-bit, 9/7 with MCT, 32x32
code-blocks; "real" is the mean over 4 frames from the middle of each of four 2K DCPs
(1998x1080), the others are `bench_decode`'s synthetic frames. Single cells vary by
up to 15% between rounds, so differences of a few percent are noise.

| Frame | Threads | Stock CPU | CPU now | Metal now | CPU now vs stock | Metal vs stock | Metal vs CPU now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real 2K, 196 Mbit/s | 1 / 4 / 8 | 147 / 50.6 / 36.0 | 141 / 37.6 / 23.0 | 40.1 / 28.8 / 22.9 | 1.05× / 1.35× / 1.57× | 3.68× / 1.76× / 1.58× | 3.52× / 1.30× / 1.00× |
| Real 2K, 150 Mbit/s | 1 / 4 / 8 | 124 / 44.9 / 32.5 | 116 / 31.6 / 19.9 | 40.4 / 29.2 / 19.9 | 1.07× / 1.42× / 1.64× | 3.06× / 1.54× / 1.63× | 2.87× / 1.08× / 1.00× |
| Real 2K, 144 Mbit/s | 1 / 4 / 8 | 115 / 41.9 / 32.3 | 109 / 29.3 / 19.8 | 40.0 / 28.5 / 18.0 | 1.06× / 1.43× / 1.63× | 2.88× / 1.47× / 1.80× | 2.72× / 1.03× / 1.10× |
| Real 2K, 105 Mbit/s | 1 / 4 / 8 | 81.7 / 35.0 / 27.9 | 74.7 / 22.1 / 14.4 | 36.9 / 22.0 / 14.5 | 1.09× / 1.58× / 1.94× | 2.22× / 1.59× / 1.93× | 2.03× / 1.00× / 0.99× |
| Synthetic 2K, 250 Mbit/s | 1 / 4 / 8 | 165 / 56.1 / 37.3 | 151 / 41.5 / 24.4 | 33.7 / 29.4 / 24.2 | 1.09× / 1.35× / 1.53× | 4.90× / 1.91× / 1.54× | 4.48× / 1.41× / 1.01× |
| Synthetic 4K, 250 Mbit/s | 1 / 4 / 8 | 252 / 130 / 115 | 206 / 65.7 / 44.6 | 120 / 65.2 / 44.0 | 1.22× / 1.99× / 2.58× | 2.11× / 2.00× / 2.61× | 1.72× / 1.01× / 1.01× |
| Synthetic 4096x2160, 250 Mbit/s | 1 / 4 / 8 | 283 / 135 / 112 | 218 / 65.9 / 44.6 | 121 / 65.6 / 45.2 | 1.30× / 2.04× / 2.52× | 2.33× / 2.05× / 2.48× | 1.80× / 1.00× / 0.99× |
| Synthetic 4096x1716, 250 Mbit/s | 1 / 4 / 8 | 238 / 107 / 88.0 | 189 / 57.8 / 36.7 | 84.4 / 54.4 / 37.0 | 1.26× / 1.86× / 2.40× | 2.82× / 1.97× / 2.38× | 2.24× / 1.06× / 0.99× |

Frames per second with several decoders in parallel (the way DCP-o-matic uses
libdcp; 20 warm-up decodes each, then a common start; the better of 2 rounds, which
differed by up to 25%):

| Frame | Decoders × threads | Stock CPU | CPU now | Metal now | CPU now vs stock | Metal vs stock | Metal vs CPU now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real 2K, 196 Mbit/s | 10 × 1 / 3 × 4 | 52.4 / 47.1 | 52.7 / 59.6 | 76.2 / 65.6 | 1.01× / 1.27× | 1.45× / 1.39× | 1.45× / 1.10× |
| Real 2K, 105 Mbit/s | 10 × 1 / 3 × 4 | 88.8 / 66.5 | 90.5 / 100 | 127 / 98.9 | 1.02× / 1.50× | 1.43× / 1.49× | 1.40× / 0.99× |
| Synthetic 4K, 250 Mbit/s | 10 × 1 / 3 × 4 | 23.8 / 17.4 | 28.3 / 31.0 | 41.3 / 31.7 | 1.19× / 1.78× | 1.74× / 1.82× | 1.46× / 1.02× |

The CPU decoder gains most with a thread pool (1.35 to 2.6×: T1 job batching, the
threaded DC level shift and, at widths of 2048 and 4096, the inverse DWT); on one
thread it gains 5 to 9% on 2K and 22 to 30% on 4K. Metal adds most where the CPU has
few threads: 1.7 to 4.5× on one decoder thread (the least on 4K) and about 1.45× with
ten single-threaded decoders in parallel; 1.0 to 1.4× at 4 threads. With a thread
pool of 8, or 3 decoders of 4 threads, the adaptive split settles on the CPU alone or
close to it, so Metal then matches the CPU decoder.

Exploring costs some time when a decoder starts: the first 50 single-threaded Metal
decodes of a real 2K frame averaged 42.8 ms (first 300: 39.6 ms, settled about
37 to 40 ms; `bench_decode`).

The inverse DWT alone, stock → now (`bench_dwt -I -iter 15`, one 9/7 plane, 6
resolutions, ms):

| Plane | 1 thread | 4 threads | 8 threads |
| --- | --- | --- | --- |
| 2048x1080 | 4.99 → 3.05 | 2.70 → 1.33 | 1.78 → 1.18 |
| 1998x1080 | 2.80 → 2.65 | 1.67 → 1.23 | 1.24 → 1.23 |
| 4096x2160 | 26.9 → 16.9 | 13.9 → 5.70 | 8.05 → 3.57 |
| 3996x2160 | 12.7 → 11.2 | 6.12 → 3.79 | 3.42 → 2.87 |
| 4096x1716 | 21.1 → 13.5 | 10.6 → 4.39 | 6.38 → 2.98 |

The GPU path keeps one batch per decode on the GPU (pool bounded by
`OPJ_METAL_T1_DEC_MAX_BATCHES`, default 5): about 33 MB for a 4K frame and 6 to 10 MB
for 2K (input bytes, descriptors and the per-code-block flags scratch; no output
buffer with zero-copy).

### Design notes

* Code-blocks are sorted by coded length, longest first, and dealt round-robin to
  32-lane SIMD groups so lanes finish together. The context states and the ZC/SC
  lookup tables live in threadgroup memory.
* The CPU's share is submitted while the GPU command buffer runs; on zero-copy the CPU
  jobs decode into private buffers and are copied into the tile after the GPU is done.
* Tried and dropped because they were slower: a column-bitmask kernel
  with all state in threadgroup memory, a state-machine kernel with a single MQ decode
  site, flags in threadgroup memory for the tail, and sorting by pass count or area.
* On real frames the GPU is throughput-bound rather than held up by the longest
  code-blocks: taking 60% of the coded bytes to the CPU hardly shortened the GPU run
  (31 to 30 ms on a 2K frame).
* GPU clocks follow the recent load. While the CPU decodes alongside, the GPU runs
  slower (a 2K frame: 29 ms with a 0% CPU share, 41 ms with 10%), and a decode right
  after a CPU-only one finds it cold; hence measurement rather than a formula, and
  warm-up decodes before a candidate's times count.
* One 2K tile leaves the GPU under-occupied: twice its code-blocks in one dispatch
  took 1.44 times as long, four times 2.35 times. Overlapping parallel decoders'
  batches (a command queue each) uses that.
* With a thread pool, CPU T1 was dominated by the pool, not by decoding: sampling
  an 8-thread 4K decode found the workers mostly in `__psynch_mutexwait`,
  `__ulock_wait2` and `malloc`/`free` (one pool job, with three allocations and two
  lock round trips, per code-block, and about 26000 mostly near-empty code-blocks).
  Batching by count alone (16) already gave most of the gain; bounding a batch by
  coded bytes as well keeps long code-blocks spread over the threads. Compiler
  flags (`-mcpu=apple-m1`, LTO) do not speed up the single-threaded T1 passes.
* The inverse 9/7 DWT was about twice as slow at power-of-two widths, all of it in
  the vertical pass (horizontal pass about equal; vertical 3.75 against 1.47 ms per
  2K plane, 20.4 against 6.6 ms per 4K plane). Upstream gathers one 8-column strip
  (32 bytes of each row) per sweep over all rows, lifts it and writes it back; at a
  row stride of 8 or 16 KB the rows share a few cache sets, so each line is evicted
  before the next strip, or the write-back, reuses it. Four strips per sweep fixed
  2K. At 4K (16 KB stride: every row in the same L1 set, and a full column more
  than its L2 sets hold) reading and writing still cost 2.7 times as much as at
  3996: software prefetch closed the gap for the reads; pipelining the lifting
  (gather 32 rows, advance each lifting step as far as its inputs allow, write back
  final rows) keeps the top half's writes next to their reads, but the bottom half
  is final only long after it was read (the transform is in place and row r's
  source sample is gathered at about position 2r). What remains is about 1.5 times
  3996's time on one thread and 1.25 on 8. Tried and dropped: 2, 8 or 16 strips per
  sweep, 8- to 64-row chunks, prefetching the write-back rows (slower), and
  `DC ZVA` before full-line writes (2%).

## Unmaintained repository

This repository should be considered unmaintained. Commits may appear from time
to time depending on committer interest and mood, but no committer currently feel
responsible to regularly review tickets or pull requests.

## What is OpenJPEG ? 

OpenJPEG is an open-source JPEG 2000 codec written in C language. It has been developed in order to promote the use of [JPEG 2000](http://www.jpeg.org/jpeg2000), a still-image compression standard from the Joint Photographic Experts Group ([JPEG](http://www.jpeg.org)).  Since April 2015, it is officially recognized by ISO/IEC and ITU-T as a [JPEG 2000 Reference Software](http://www.itu.int/rec/T-REC-T.804-201504-I!Amd2).

## Who can use the code ?
[![badge-license]][link-license]

Anyone. As the OpenJPEG code is released under the [BSD 2-clause "Simplified" License][link-license], anyone can use or modify the code, even for commercial applications. The only restriction is to retain the copyright in the sources or in the binaries documentation. Of course, if you modified the code in a way that might be of interest for other users, you are encouraged to share it (through a [github pull request](https://github.com/uclouvain/openjpeg/pulls) or by filling an [issue](https://github.com/uclouvain/openjpeg/issues)) but this is not a requirement.

## How to install and use OpenJPEG ?
API Documentation needs a major refactoring. Meanwhile, you can check [installation](https://github.com/uclouvain/openjpeg/wiki/Installation) instructions and [codec documentation](https://github.com/uclouvain/openjpeg/wiki/DocJ2KCodec).
    
## Current Status
[![badge-build]][link-build]

[![badge-msvc-build]][link-msvc-build] 

[![badge-coverity]][link-coverity]

## Who are the developers ?

The library is developed and maintained by the Image and Signal Processing Group ([ISPGroup](http://sites.uclouvain.be/ispgroup/)), in the Université catholique de Louvain ([UCL](http://www.uclouvain.be/en-index.html), with the support of the [CNES](https://cnes.fr/), the [CS](http://www.c-s.fr/) company and the [intoPIX](http://www.intopix.com) company. The JPWL module has been developed by the Digital Signal Processing Lab ([DSPLab](http://dsplab.diei.unipg.it/)) of the University of Perugia, Italy ([UNIPG](http://www.unipg.it/)).

## Details on folders hierarchy

* src
  * lib
    * openjp2: contains the sources of the openjp2 library (Part 1 & 2)
    * openjpip: complete client-server architecture for remote browsing of jpeg 2000 images.
  * bin: contains all applications that use the openjpeg library
    * common: common files to all applications
    * jp2: a basic codec
    * jpip: OpenJPIP applications (server and dec server)
      * java: a Java client viewer for JPIP
    * wx
      * OPJViewer: gui for displaying j2k files (based on wxWidget)
* wrapping
  * java: java jni to use openjpeg in a java program
* thirdparty: thirdparty libraries used by some applications. These libraries will be built only if there are not found on the system. Note that libopenjpeg itself does not have any dependency.
* doc: doxygen documentation setup file and man pages
* tests: configuration files and utilities for the openjpeg test suite. All test images are located in [openjpeg-data](https://github.com/uclouvain/openjpeg-data) repository.
* cmake: cmake related files
* scripts: scripts for developers

See [LICENSE][link-license] for license and copyright information.

See [INSTALL](https://github.com/uclouvain/openjpeg/blob/master/INSTALL.md) for installation procedures.

See [NEWS](https://github.com/uclouvain/openjpeg/blob/master/NEWS.md) for user visible changes in successive releases.

## Deprecated components

The openjpwl, openjp3d and openmj2 components have been removed after the
2.4.0 release. Their building and working state is unknown. People interested
in them should start from the 2.4.0 tag.

## API/ABI

An API/ABI timeline is automatically updated [here][link-api-timeline].

OpenJPEG strives to provide a stable API/ABI for your applications. As such it
only exposes a limited subset of its functions.  It uses a mechanism of
exporting/hiding functions. If you are unsure which functions you can use in
your applications, you should compile OpenJPEG using something similar to gcc:
`-fvisibility=hidden` compilation flag.
See also: http://gcc.gnu.org/wiki/Visibility

On windows, MSVC directly supports export/hiding function and as such the only
API available is the one supported by OpenJPEG.

[comment-license]: https://img.shields.io/github/license/uclouvain/openjpeg.svg "https://img.shields.io/badge/license-BSD--2--Clause-blue.svg"
[badge-license]: https://img.shields.io/badge/license-BSD--2--Clause-blue.svg "BSD 2-clause \"Simplified\" License"
[link-license]: https://github.com/uclouvain/openjpeg/blob/master/LICENSE "BSD 2-clause \"Simplified\" License"
[badge-build]: https://github.com/uclouvain/openjpeg/actions/workflows/build.yml/badge.svg "Build Status"
[link-build]: https://github.com/uclouvain/openjpeg/actions/workflows/build.yml "Build Status"
[badge-msvc-build]: https://ci.appveyor.com/api/projects/status/github/uclouvain/openjpeg?branch=master&svg=true "Windows Build Status"
[link-msvc-build]: https://ci.appveyor.com/project/detonin/openjpeg/branch/master "Windows Build Status"
[badge-coverity]: https://scan.coverity.com/projects/6383/badge.svg "Coverity Scan Build Status"
[link-coverity]: https://scan.coverity.com/projects/uclouvain-openjpeg "Coverity Scan Build Status"
[link-api-timeline]: http://www.openjpeg.org/abi-check/timeline/openjpeg "OpenJPEG API/ABI timeline"
