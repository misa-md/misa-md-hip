//
// Created by genshen on 2021/5/21.
//

#ifndef MISA_MD_HIP_DF_DOUBLE_BUFFER_IMP_H
#define MISA_MD_HIP_DF_DOUBLE_BUFFER_IMP_H

#include <hip/hip_runtime.h>

#include "../kernel_types.h"
#include "atom/atom_element.h"
#include "double_buffer.h"
#include "double_buffer_base_imp.hpp"
#include "hip_kernels.h"

/**
 * double buffer implementation for calculating derivative of embedded energy: df.
 */
class DfDoubleBufferImp : public DoubleBufferBaseImp<_cuAtomElement, AtomElement, AtomElement> {
public:
  /**
   * All parameters are the same as rho calculation.
   * @param stream1,stream2 2 hip streams used for syncing buffer 1 and buffer 2 (e.g. data copying).
   * @param blocks total blocks for double buffer.
   * @param data_len total z-layers of atoms on current processor (not include ghost regions).
   * @param _ptr_atoms atoms data in host side.
   * @param _ptr_device_buf1, _ptr_device_buf2 2 buffer memory on device side
   * @param _d_dfs the results data on device side (calculating results will writ to this array).
   * @param h_domain domain information
   */
  DfDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len,
                    AtomElement *_ptr_atoms, _cuAtomElement *_ptr_device_buf1, _cuAtomElement *_ptr_device_buf2,
                    tp_device_rho *_d_dfs, _hipDeviceDomain h_domain);

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const int block_id) override;

private:
  // lattice atoms array in current MPI process (including ghost regions)
  AtomElement *ptr_atoms = nullptr;
  tp_device_rho *d_dfs = nullptr;
  const _hipDeviceDomain h_domain;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;
};

#endif // MISA_MD_HIP_DF_DOUBLE_BUFFER_IMP_H
