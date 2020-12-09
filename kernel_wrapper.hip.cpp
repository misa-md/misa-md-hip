#include "kernel_wrapper.h"
#include <hip/hip_runtime.h>

void __kernel_calRho_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                             double cutoff_radius) {
  hipLaunchKernelGGL(calRho, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, offsets, cutoff_radius);
}

void __kernel_calDf_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets) {
  hipLaunchKernelGGL(calDf, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, offsets);
}

void __kernel_calForce_wrapper(dim3 grid_dims, dim3 blocks_dims, _cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets,
                               double cutoff_radius) {
  hipLaunchKernelGGL(calForce, dim3(grid_dims), dim3(blocks_dims), 0, 0, d_atoms, offsets, cutoff_radius);
}
