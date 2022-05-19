#!/bin/sh

WRAPPER_PATH=cmd-wrapper
HIPCC_PATH=/opt/compilers/rocm/4.2.0/bin/hipcc

export WRAPPED_CMD=$HIPCC_PATH
export WRAPPED_REMOVE_DUP_ARGS=../src/arch_hip/lib/libmd_arch_hip.a:../lib/libmd.a:../src/arch_hip/lib/libmd_arch_hip_normal.a:../../lib/libmd.a:../../src/arch_hip/lib/libmd_arch_hip.a:../../src/arch_hip/lib/libmd_arch_hip_normal.a

$WRAPPER_PATH $@
