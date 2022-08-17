//
// Created by genshen on 2020/04/27.
//

#include <hip/hip_runtime.h>

#include "kernel_wrapper.h"

void __kernel_calRho_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                             double cutoff_radius) {
  hipLaunchKernelGGL(calc_rho, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, nullptr, offsets, 0, 0,
                     cutoff_radius);
}

void __kernel_calDf_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms) {
  hipLaunchKernelGGL(cal_df_aos, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, 0, 0);
}

void __kernel_calForce_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                               double cutoff_radius) {
  hipLaunchKernelGGL(calForce, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, offsets, cutoff_radius);
}
