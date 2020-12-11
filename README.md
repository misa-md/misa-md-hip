# MISA-MD-hip
hip implementation of MISA-MD for nvidia GPU, AMD GPU and DCU.

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
    +   git.hpcer.dev/HPCer/MISA-MD/hip-potential@v0.1.0@hip_pot:
    +     features: ["HIP_POT_TEST_BUILD_ENABLE_FLAG=OFF"]
        github.com/Taywee/args@6.2.2@args: {build: ["CP args.hxx {{.INCLUDE}}/args.hpp"]}
    ```
    or:
    ```diff
    dependencies:
    packages:
    +   github.com/misa-md/hip-potential@v0.1.0@hip_pot:
    +     features: ["HIP_POT_TEST_BUILD_ENABLE_FLAG=OFF"]
        github.com/Taywee/args@6.2.2@args: {build: ["CP args.hxx {{.INCLUDE}}/args.hpp"]}
    ```

3. then build dependdenccies by running:
    ```bash
    pkg fetch
    pkg install
    ```

4. build MISA-MD for hip:

    ```bash
    # in MISA-MD directory
    cmake -B./cmake-build-hip -S./  \
        -DMD_HIP_ARCH_ENABLE_FLAG=ON \
        -DMD_HIP_ARCH_SRC_PATH=../misa-md-hip \
        -DCMAKE_LINKER=hipcc \
        -DCMAKE_CXX_LINK_EXECUTABLE="<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
    cmake --build ./cmake-build-hip/ -j 4
    ```
   Note: If you need to generate an optimized executable file,
   you can use flag `-DCMAKE_BUILD_TYPE=Release` in cmake configuration step,
   which will use `-O3` flag for compling and linking. 
