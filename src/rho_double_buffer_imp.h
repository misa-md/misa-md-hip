//
// Created by genshen on 2021/5/18.
//

#ifndef MISA_MD_RHO_DOUBLE_BUFFER_IMP_H
#define MISA_MD_RHO_DOUBLE_BUFFER_IMP_H

#include <hip/hip_runtime.h>

#include "atom/atom_element.h"
#include "double_buffer.h"
#include "hip_kernels.h"
#include "global_ops.h"

/**
 * double buffer implementation for calculating electron density rho.
 */
class RhoDoubleBufferImp : public DoubleBuffer {
public:
  /**
   * @param stream1 stream for buffer 1, which is used for syncing buffer 1.
   * @param stream2 stream for buffer 2, which is used for syncing buffer 2.
   * @param blocks total data blocks
   * @param data_len total data length in all data blocks
   * @param _ptr_atoms atoms pointer on host side
   * @param _ptr_device_buf1, _ptr_device_buf2 two atom buffers memory on device side.
   * @param h_domain domain information
   * @param d_nei_offset neighbor offset data
   * @param cutoff_radius cutoff
   */
  RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len,
                     AtomElement *_ptr_atoms, _cuAtomElement *_ptr_device_buf1, _cuAtomElement *_ptr_device_buf2,
                     tp_device_rho *_d_rhos, _hipDeviceDomain h_domain, const _hipDeviceNeiOffsets d_nei_offset,
                     const double cutoff_radius);

  /**
   * implementation of copying data into device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param data_start_index, data_end_index data starting and ending index(not include ending index)
   *   for current data block.
   * @param block_id current data block id.
   */
  void fillBuffer(hipStream_t &stream, const bool left, const unsigned int data_start_index,
                  const unsigned int data_end_index, const int block_id) override;

  /**
   * implementation of fetching data from device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param data_start_index, data_end_index data starting and ending index(not include ending index)
   *   for current data block.
   * @param block_id current data block id.
   */
  void fetchBuffer(hipStream_t &stream, const bool left, const unsigned int data_start_index,
                   const unsigned int data_end_index, const int block_id) override;

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const int block_id) override;

private:
  // lattice atoms array in current MPI process (including ghost regions)
  AtomElement *ptr_atoms = nullptr;
  _cuAtomElement *_ptr_device_buf1 = nullptr, *_ptr_device_buf2 = nullptr;
  tp_device_rho *d_rhos = nullptr;
  const _hipDeviceDomain h_domain;
  const _hipDeviceNeiOffsets d_nei_offset; // fixme: remove it
  const double cutoff_radius;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;
};

#endif // MISA_MD_RHO_DOUBLE_BUFFER_IMP_H
