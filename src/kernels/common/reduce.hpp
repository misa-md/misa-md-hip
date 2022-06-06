//
// Created by genshen on 2022/6/5.
//

#ifndef MISA_MD_HIP_REDUCE_HPP
#define MISA_MD_HIP_REDUCE_HPP

#include <hip/hip_runtime.h>

// register and __shfl_down based wavefront reduction
template <typename T, int WF_SIZE>
__device__ __forceinline__ void SHFL_DOWN_WF_REDUCE(T &total_sum, const T local_sum) {
  total_sum += local_sum;
  if (WF_SIZE > 32) {
    total_sum += __shfl_down(total_sum, 32, WF_SIZE);
  }
  if (WF_SIZE > 16) {
    total_sum += __shfl_down(total_sum, 16, WF_SIZE);
  }
  if (WF_SIZE > 8) {
    total_sum += __shfl_down(total_sum, 8, WF_SIZE);
  }
  if (WF_SIZE > 4) {
    total_sum += __shfl_down(total_sum, 4, WF_SIZE);
  }
  if (WF_SIZE > 2) {
    total_sum += __shfl_down(total_sum, 2, WF_SIZE);
  }
  if (WF_SIZE > 1) {
    total_sum += __shfl_down(total_sum, 1, WF_SIZE);
  }
}

#endif // MISA_MD_HIP_REDUCE_HPP
