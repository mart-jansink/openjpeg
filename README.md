# OPENJPEG Library and Applications

## CtrlMovie fork: Metal-accelerated encoder for Apple Silicon

This repository is CtrlMovie AG's fork of OpenJPEG 2.5.4, branch `metal-encoder`.
It adds a GPU (Metal) path to the JPEG 2000 **encoder** for Cinema 2K/4K DCP export
on Apple Silicon, plus CPU improvements that keep the codestream byte-identical to
upstream OpenJPEG. The decoder is unchanged. Copyright (c) 2026, CtrlMovie AG on the
changed and added files; the BSD 2-clause license is unchanged.

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
