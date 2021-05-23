cmake_minimum_required(VERSION 3.8)

project(MISA-MD-hip CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

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


#set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

set(MD_HIP_ARCH_FILES
        hip_arch_acc_imp.cpp
        src/double-buffer/double_buffer.cpp
        src/double-buffer/rho_double_buffer_imp.cpp
        src/double-buffer/df_double_buffer_imp.cpp
        src/global_ops.cpp
        # hip_kernels.h
        hip_kernels.hip.cpp
#        kernel_wrapper.h
        kernel_wrapper.hip.cpp
        )
set_source_files_properties(${MD_HIP_ARCH_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(MD_HIP_LIB_NAME md_arch_hip)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

hip_add_library(${MD_HIP_LIB_NAME} STATIC ${MD_HIP_ARCH_FILES})
# target_compile_options(${MD_HIP_LIB_NAME} PRIVATE "-Xcompiler" "-fPIC")

target_include_directories(
        ${MD_HIP_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        ${MD_HIP_LIB_NAME}
        PUBLIC
        md
        hip_pot::hip_pot
        hip_hcc  # hip lib for dcu.
)
