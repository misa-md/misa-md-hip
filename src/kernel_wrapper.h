//
// Created by genshen on 2020/04/27.
//

#ifndef HIP_KERNEL_WRAPPER_H
#define HIP_KERNEL_WRAPPER_H

#include "kernels/hip_kernels.h"

// wrapper function for launching kernel function.
void __kernel_calRho_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                             double cutoff_radius);

void __kernel_calDf_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms);

void __kernel_calForce_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                               double cutoff_radius);

#endif // HIP_KERNEL_WRAPPER_H
