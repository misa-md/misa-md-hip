//
// Created by genshen on 2022/8/16.
//

#ifndef MISA_MD_KERNEL_UTILS_H
#define MISA_MD_KERNEL_UTILS_H

#include <hip/hip_runtime.h>

#include "md_hip_building_config.h"
#include "md_hip_config.h"

#if __CUDA_ARCH__ < 600
__device__ inline float atomicAdd_(float *address, float val) { atomicAdd(address, val); }

__device__ inline double atomicAdd_(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#else
template <typename T> __device__ __forceinline__ T atomicAdd_(T *a, T b) { return atomicAdd(a, b); }
#endif

template <typename T> __device__ __forceinline__ void hip_md_interaction_add(T *addr, const T value) {
  if (global_config::use_newtons_third_law()) {
    atomicAdd_(addr, value);
  } else {
    *addr += value;
  }
}

#endif // MISA_MD_KERNEL_UTILS_H
