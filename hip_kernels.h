//
// Created by genshen on 2020/04/27.
//

#ifndef HIP_KERNELS_H
#define HIP_KERNELS_H

#include "src/global_ops.h"
#include <hip/hip_runtime.h>

/**
 * calculate rho on device side
 * @param d_atoms subset of atoms (including ghost)
 * @param _d_rhos rho array for storing result
 * @param offsets offset array for neighbor searching
 * @param start_id start index
 * @param end_id end index
 * @param cutoff_radius cutoff
 */
__global__ void calc_rho(_cuAtomElement *d_atoms, tp_device_rho *_d_rhos, _hipDeviceNeiOffsets offsets,
                         const _type_atom_index_kernel start_id, const _type_atom_index_kernel end_id,
                         double cutoff_radius);

__global__ void calDf(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets);

__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius);

__device__ inline void atomicAdd_(double *a, double b) { *a += b; }

#endif // HIP_KERNELS_H
