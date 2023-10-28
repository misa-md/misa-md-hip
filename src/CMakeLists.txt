# see https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable
if (NOT DEFINED HIP_PATH)
    if (NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else ()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif ()
endif ()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)
if (HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else ()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif ()

include(configure.cmake)

#set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

set(MD_HIP_ARCH_NORMAL_FILES
        cli.cpp
        )

hip_add_library(md_arch_hip_normal STATIC ${MD_HIP_ARCH_NORMAL_FILES})
target_include_directories(
        md_arch_hip_normal
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(md_arch_hip_normal PUBLIC md)

## set hip side source files.
set(MD_HIP_ARCH_HEADER_FILES
        kernels/hip_kernels.h
        kernel_wrapper.h
        memory/device_atoms.h
        kernels/atom_index.hpp
        kernels/soa-block-atom/soa_block_atom.hpp
        kernels/soa-thread-atom/soa_thread_atom.h
        kernels/soa-wf-atom/soa_wf_atom.h
        kernels/bitmap/bitmap_soa_thread_atom.h
        )

set(MD_HIP_ARCH_SOURCE_FILES
        double-buffer/double_buffer.cpp
        double-buffer/rho_double_buffer_imp.cpp
        double-buffer/df_double_buffer_imp.cpp
        double-buffer/force_double_buffer_imp.cpp
        global_ops.cpp
        hip_arch_acc_imp.cpp
        kernels/hip_kernels.hip.cpp
        kernels/soa-thread-atom/soa_thread_atom.cpp
        kernels/soa-wf-atom/soa_wf_atom.cpp
        kernel_wrapper.hip.cpp
        memory/device_atoms_aos.cpp
        memory/device_atoms_soa.cpp
        mem.cpp
        double-buffer/bitmap_double_buffer_imp.cpp
        kernels/bitmap/bitmap_soa_thread_atom.cpp
        )

set_source_files_properties(${MD_HIP_ARCH_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(MD_HIP_LIB_NAME md_arch_hip)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

hip_add_library(${MD_HIP_LIB_NAME} STATIC ${MD_HIP_ARCH_SOURCE_FILES} ${MD_HIP_ARCH_HEADER_FILES})
# target_compile_options(${MD_HIP_LIB_NAME} PRIVATE "-Xcompiler" "-fPIC")

#set_target_properties(${MD_HIP_LIB_NAME} PROPERTIES LINKER_LANGUAGE HIP)

target_include_directories(
        ${MD_HIP_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)
# set_property(TARGET ${MD_HIP_LIB_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# MESSAGE(STATUS ${CUDA_LIBRARIES})

target_link_libraries(
        ${MD_HIP_LIB_NAME}
        PUBLIC
        md
        hip_pot::hip_pot
        md_arch_hip_normal
)
