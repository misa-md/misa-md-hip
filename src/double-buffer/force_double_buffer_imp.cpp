//
// Created by genshen on 2021/5/23.
//

#include <hip/hip_runtime.h>
#include <iostream>

#include "atom/atom_element.h"
#include "hip_macros.h" // from hip_pot lib

#include "../kernel_itl.hpp"
#include "force_double_buffer_imp.h"
#include "hip_kernels.h"
#include "md_hip_config.h"

ForceDoubleBufferImp::ForceDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks,
                                           const unsigned int data_len, AtomElement *_ptr_atoms,
                                           _cuAtomElement *_ptr_device_buf1, _cuAtomElement *_ptr_device_buf2,
                                           tp_device_force *_d_forces, _hipDeviceDomain h_domain,
                                           const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius)
    : DoubleBufferBaseImp(stream1, stream2, blocks, data_len, 0, 0, 0, 0, _ptr_atoms, _ptr_atoms, nullptr,
                          _ptr_device_buf1, _ptr_device_buf2),
      ptr_atoms(_ptr_atoms), d_forces(_d_forces), h_domain(h_domain), d_nei_offset(d_nei_offset),
      cutoff_radius(cutoff_radius), atoms_per_layer(h_domain.ext_size_x * h_domain.ext_size_y) {
  constexpr int threads_per_block = 256;
  this->kernel_config_block_dim = dim3(threads_per_block);

  // One thread only process one atom.
  // note: size_x in h_domain is double
  const _type_atom_count local_atoms_num = h_domain.box_size_x * h_domain.box_size_y * h_domain.box_size_z;
  int blocks_num = (local_atoms_num - 1) / threads_per_block + 1;
  this->kernel_config_grid_dim = dim3(blocks_num);
}

void ForceDoubleBufferImp::calcAsync(hipStream_t &stream, const int block_id) {
  unsigned int data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  _cuAtomElement *d_p = block_id % 2 == 0 ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  d_p += atoms_per_layer * h_domain.ghost_size_z;
  tp_device_force *force_ptr = d_forces + atoms_per_layer * (data_start_index + h_domain.ghost_size_z);
  // atoms number to be calculated in this block
  const std::size_t atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);
  (itl_atoms_pair<tp_device_force, ModeForce>)<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(
      d_p, force_ptr, d_nei_offset, data_start_index, data_end_index, cutoff_radius);
}
