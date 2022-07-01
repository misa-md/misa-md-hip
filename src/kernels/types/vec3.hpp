//
// Created by genshen on 2022/6/5.
//

#ifndef MISA_MD_HIP_VEC3_HPP
#define MISA_MD_HIP_VEC3_HPP

#include <hip/hip_runtime.h>

#include "kernels/common/reduce.hpp"
#include "kernels/hip_kernel_types.h"
#include "md_hip_building_config.h"

template <typename T> struct _type_vec1 {
  T data = 0.0;

  __device__ __forceinline__ void set_v(const _type_vec1<T> vec1) { data = vec1.data; }

  template <typename I> __device__ __forceinline__ void store_to(_type_vec1<T> *ptr, I offset) { ptr[offset] = *this; }

  __device__ __forceinline__ T first() { return data; }

  __device__ __forceinline__ void wf_reduce() {
    T t0_sum = 0.0;
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t0_sum, data);
    data = t0_sum;
  }
};

template <typename T> struct _type_vec3 {
  T data[3] = {0.0, 0.0, 0.0};
  __device__ __forceinline__ void set_v(const _type_vec3<T> vec3) {
    data[0] = vec3.data[0];
    data[1] = vec3.data[1];
    data[2] = vec3.data[2];
  }

  template <typename I> __device__ __forceinline__ void store_to(_type_vec3<T> *ptr, I offset) { ptr[offset] = *this; }

  __device__ __forceinline__ T first() { return data[0]; };

  __device__ __forceinline__ void wf_reduce() {
    T t0_sum = 0.0, t1_sum = 0.0, t2_sum = 0.0;
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t0_sum, data[0]);
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t1_sum, data[1]);
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t2_sum, data[2]);
    data[0] = t0_sum;
    data[1] = t1_sum;
    data[2] = t2_sum;
  }
};

template struct _type_vec1<float>;
typedef _type_vec1<float> _type_s_vec1;

template struct _type_vec1<double>;
typedef _type_vec1<double> _type_d_vec1;

template struct _type_vec3<float>;
typedef _type_vec3<float> _type_s_vec3;

template struct _type_vec3<double>;
typedef _type_vec3<double> _type_d_vec3;

#endif // MISA_MD_HIP_VEC3_HPP
