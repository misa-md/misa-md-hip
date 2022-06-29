# MISA-MD-hip
hip implementation of MISA-MD for nvidia GPU, AMD GPU and DCU.

## Requirements
- HIP 3.5 or above
- MPI
- ROCm (AMD GPU) or CUDA (NVIDIA GPU)
- [pkg](https://github.com/genshen/pkg) and [cmd-wrapper](https://github.com/genshen/cmd-wrapper). 

For pkg and cmd-wrapper: just download the binary release and put them into directory specified by `$PATH` env.

## Build
1. get MISA-MD source code.
    We assume following directory structure and we are in `MISA-MD` directory
    (`misa-md-hip` is current repository):
    ```
    |-- MISA-MD
    |-- misa-md-hip
    ```
2. edit file `pkg.yaml` to add `hip-potential` package:

    ```diff
    dependencies:
    packages:
    +   git.hpcer.dev/HPCer/MISA-MD/hip-potential@v0.3.0@hip_pot:
    +     features: ["HIP_POT_TEST_BUILD_ENABLE_FLAG=OFF", "CMAKE_CXX_FLAGS=-fPIC", "HIP_HIPCC_FLAGS=\"-fgpu-rdc -std=c++11\"", "HIP_NVCC_FLAGS=-rdc=true" ]
    ```
    or:
    ```diff
    dependencies:
    packages:
    +   github.com/misa-md/hip-potential@v0.3.0@hip_pot:
    +     features: ["HIP_POT_TEST_BUILD_ENABLE_FLAG=OFF", "CMAKE_CXX_FLAGS=-fPIC", "HIP_HIPCC_FLAGS=\"-fgpu-rdc -std=c++11\"", "HIP_NVCC_FLAGS=-rdc=true" ]
        github.com/Taywee/args@6.2.2@args: {build: ["CP args.hxx {{.INCLUDE}}/args.hpp"]}
    ```

3. then build dependdenccies by running:
    ```bash
    pkg fetch
    pkg install
    ```

4. build MISA-MD for hip (ROCm platform):
    check HIP_PATH and other config in script `scripts/hipcc-wrapper-rocm.sh`
    ```bash
    # in MISA-MD directory
    export CC=clang
    export CXX=${PWD}/scripts/hipcc-wrapper-rocm.sh
    cmake -B./cmake-build-hip -S./  \
        -DMD_HIP_ARCH_ENABLE_FLAG=ON \
        -DMD_HIP_ARCH_SRC_PATH=../misa-md-hip
    cmake --build ./cmake-build-hip/ -j 4
    ```

    build MISA-MD for hip (NVIDIA platform):
    check HIP_PATH and other config in script `scripts/hipcc-wrapper-nv.sh`, then run:
    ```bash
    export CC=clang
    export CXX=clang++
    LINKER=${PWD}/scripts/hipcc-wrapper.sh

    cmake -B ./build-gpu-test -S./ \
      -DMD_HIP_ARCH_ENABLE_FLAG=ON -DMD_HIP_ARCH_SRC_PATH=../misa-md-hip \
      -DCMAKE_BUILD_TYPE=Release \
      -DHIP_HIPCC_FLAGS="-std=c++11" \
      -DHIP_NVCC_FLAGS="-rdc=true" \
      -DCMAKE_LINKER=$LINKER \
      -DCMAKE_CXX_LINK_EXECUTABLE="<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
   ```
   Note: If you need to generate an optimized executable file,
   you can use flag `-DCMAKE_BUILD_TYPE=Release` in cmake configuration step,
   which will use `-O3` flag for compling and linking. 
