#!/bin/sh

# use cmd-wrapper: https://github.com/genshen/cmd-wrapper (version: 6ac1cd1)
WRAPPER_PATH=cmd-wrapper
HIPCC_PATH=/public/software/compiler/dtk/dtk-22.10.1/hip/bin/hipcc # you may need to change the hipcc path.
LINK_FLAGS=-fgpu-rdc

export WRAPPED_CMD=$HIPCC_PATH
export WRAPPED_REMOVE_DUP_ARGS=../src/arch_hip/lib/libmd_arch_hip.a:../lib/libmd.a:../src/arch_hip/lib/libmd_arch_hip_normal.a:../../lib/libmd.a:../../src/arch_hip/lib/libmd_arch_hip.a:../../src/arch_hip/lib/libmd_arch_hip_normal.a
export WRAPPED_PREPEND_ARGS=$LINK_FLAGS
export WRAPPED_PREPEND_IF="(\\S+)libmd_arch_hip\\.a\$"

$WRAPPER_PATH $@
