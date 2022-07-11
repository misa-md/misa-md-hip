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

  template <int THREADS_IN_BLOCK, int WAVES_IN_BLOCK>
  __device__ __forceinline__ void block_wf_reduce(const int tid_in_wf, const int tid_in_block,
                                                  const int wave_id_in_block, _type_vec1<T> *temp_mem) {
    wf_reduce();
    // reduction in shared memory
    if (tid_in_wf == 0) {
      temp_mem[wave_id_in_block].data = this->data;
    }

    data = 0.0;
    __syncthreads(); // wait shared memory writing finishes.
    if (wave_id_in_block == 0) {
#pragma unroll
      for (int i = tid_in_block; i < WAVES_IN_BLOCK; i += __WAVE_SIZE__) {
        data += temp_mem[i].data;
      }
      wf_reduce();
    }
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

  //  _type_vec3<T> &operator=(const _type_vec3<T> &other) {
  //
  //  }

  __device__ __forceinline__ void wf_reduce() {
    T t0_sum = 0.0, t1_sum = 0.0, t2_sum = 0.0;
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t0_sum, data[0]);
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t1_sum, data[1]);
    SHFL_DOWN_WF_REDUCE<T, __WAVE_SIZE__>(t2_sum, data[2]);
    data[0] = t0_sum;
    data[1] = t1_sum;
    data[2] = t2_sum;
  }

  template <int THREADS_IN_BLOCK, int WAVES_IN_BLOCK>
  __device__ __forceinline__ void block_wf_reduce(const int tid_in_wf, const int tid_in_block,
                                                  const int wave_id_in_block, _type_vec3<T> *temp_mem) {
    wf_reduce();
    // reduction in shared memory
    if (tid_in_wf == 0) {
      temp_mem[wave_id_in_block].data[0] = this->data[0];
      temp_mem[wave_id_in_block].data[1] = this->data[1];
      temp_mem[wave_id_in_block].data[2] = this->data[2];
    }

    this->data[0] = 0.0;
    this->data[1] = 0.0;
    this->data[2] = 0.0;
    __syncthreads(); // wait shared memory writing finishes.
    if (wave_id_in_block == 0) {
#pragma unroll
      for (int i = tid_in_block; i < WAVES_IN_BLOCK; i += __WAVE_SIZE__) {
        data[0] += temp_mem[i].data[0];
        data[1] += temp_mem[i].data[1];
        data[2] += temp_mem[i].data[2];
      }
      wf_reduce();
    }
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
