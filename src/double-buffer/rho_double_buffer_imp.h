//
// Created by genshen on 2021/5/18.
//

#ifndef MISA_MD_HIP_RHO_DOUBLE_BUFFER_IMP_H
#define MISA_MD_HIP_RHO_DOUBLE_BUFFER_IMP_H

#include <hip/hip_runtime.h>

#include "../global_ops.h"
#include "atom/atom_element.h"
#include "double_buffer.h"
#include "double_buffer_base_imp.hpp"
#include "kernels/hip_kernels.h"
#include "memory/device_atoms.h"

typedef device_atoms::_type_buffer_desc type_rho_buffer_desc;
typedef _type_atom_list_collection type_rho_src_desc;
typedef _type_atom_list_collection type_rho_dest_desc;

/**
 * double buffer implementation for calculating electron density rho.
 */
class RhoDoubleBufferImp
    : public DoubleBufferBaseImp<type_rho_buffer_desc, type_rho_src_desc, type_rho_dest_desc> {
public:
  /**
   * @param stream1 stream for buffer 1, which is used for syncing buffer 1.
   * @param stream2 stream for buffer 2, which is used for syncing buffer 2.
   * @param blocks total data blocks
   * @param data_len total data length in all data blocks
   * @param _ptr_atoms atoms pointer descriptor on host side
   * @param _ptr_device_buf1, _ptr_device_buf2 two atom buffers descriptor memory on device side.
   * @param h_domain domain information
   * @param d_nei_offset neighbor offset data
   * @param cutoff_radius cutoff
   */
  RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len,
                     type_rho_src_desc _ptr_atoms, type_rho_buffer_desc _ptr_device_buf1,
                     type_rho_buffer_desc _ptr_device_buf2, _hipDeviceDomain h_domain,
                     const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius);

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const int block_id) override;

private:
  // lattice atoms array in current MPI process (including ghost regions)
  type_rho_src_desc ptr_atoms;

  const _hipDeviceDomain h_domain;
  const _hipDeviceNeiOffsets d_nei_offset; // fixme: remove it
  const double cutoff_radius;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;

private:
  void copyFromHostToDeviceBuf(hipStream_t &stream, type_rho_buffer_desc dest_ptr, type_rho_src_desc src_ptr,
                               const std::size_t src_offset, std::size_t size) override;

  void copyFromDeviceBufToHost(hipStream_t &stream, type_rho_dest_desc dest_ptr, type_rho_buffer_desc src_ptr,
                               const std::size_t src_offset, const std::size_t des_offset, std::size_t size) override;
};

#endif // MISA_MD_HIP_RHO_DOUBLE_BUFFER_IMP_H
