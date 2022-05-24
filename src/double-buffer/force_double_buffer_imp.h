//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H
#define MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H

#include <hip/hip_runtime.h>

#include "atom/atom_element.h"
#include "double_buffer.h"
#include "double_buffer_base_imp.hpp"
#include "../global_ops.h"
#include "kernels/hip_kernels.h"

class ForceDoubleBufferImp : public DoubleBufferBaseImp<_cuAtomElement, AtomElement, AtomElement> {
public:
  ForceDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks,
                       const unsigned int data_len, AtomElement *_ptr_atoms, _cuAtomElement *_ptr_device_buf1,
                       _cuAtomElement *_ptr_device_buf2, tp_device_force *_d_forces, _hipDeviceDomain h_domain,
                       const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius);

  void calcAsync(hipStream_t &stream, const int block_id) override;

private:
  // lattice atoms array in current MPI process (including ghost regions)
  AtomElement *ptr_atoms = nullptr;
  tp_device_force *d_forces = nullptr;
  const _hipDeviceDomain h_domain;
  const _hipDeviceNeiOffsets d_nei_offset; // fixme: remove it
  const double cutoff_radius;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;
};

#endif // MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H
