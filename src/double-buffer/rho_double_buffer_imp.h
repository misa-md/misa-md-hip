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
typedef device_atoms::_type_atom_list_desc type_rho_src_desc;
typedef device_atoms::_type_atom_list_desc type_rho_dest_desc;

typedef device_atoms::_type_dev_buffer_aos type_rho_buffer_aos_desc;
typedef device_atoms::_type_dev_buffer_soa type_rho_buffer_soa_desc;

typedef device_atoms::_type_atom_list_soa type_rho_src_soa_desc;
typedef device_atoms::_type_atom_list_aos type_rho_src_aos_desc;
typedef device_atoms::_type_atom_list_soa type_rho_dest_soa_desc;
typedef device_atoms::_type_atom_list_aos type_rho_dest_aos_desc;

/**
 * double buffer implementation for calculating electron density rho.
 */
class RhoDoubleBufferImp : public DoubleBufferBaseImp<type_rho_buffer_desc, type_rho_src_desc, type_rho_dest_desc> {
public:
  /**
   * @param stream1 stream for buffer 1, which is used for syncing buffer 1.
   * @param stream2 stream for buffer 2, which is used for syncing buffer 2.
   * @param data_desc the descriptor of source data
   * @param src_atoms_desc the host side atoms descriptor when transferring atoms data from host side to device side.
   *    For AoS memory layout, it can be the lattice atoms array in current MPI process (including ghost regions).
   * @param dest_atoms_desc the host side atoms descriptor when transferring atoms data from device side to host side.
   *    For AoS memory layout, it can keep the same as @param src_atoms_desc.
   * @param _ptr_device_buf1, _ptr_device_buf2 two atom buffers descriptor memory on device side.
   * @param h_domain domain information
   * @param d_nei_offset neighbor offset data
   * @param cutoff_radius cutoff
   */
  RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const db_buffer_data_desc data_desc,
                     type_rho_src_desc src_atoms_desc, type_rho_dest_desc dest_atoms_desc,
                     type_rho_buffer_desc _ptr_device_buf1, type_rho_buffer_desc _ptr_device_buf2,
                     _hipDeviceDomain h_domain, const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius);

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) override;

private:
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

private:
  // copy atoms data from host side to device buffer, where the memory layout of the host atoms and buffer is AoS.
  void copyHostToDevBuf_AoS(hipStream_t &stream, type_rho_buffer_aos_desc dest_ptr, type_rho_src_aos_desc src_ptr,
                            const std::size_t src_offset, std::size_t size);
  // similar as above, but the memory layout is SoA.
  void copyHostToDevBuf_SoA(hipStream_t &stream, type_rho_buffer_soa_desc dest_ptr, type_rho_src_soa_desc src_ptr,
                            const std::size_t src_offset, std::size_t size);

  // copy atoms data from device buffer side to host side, where the memory layout of the host atoms and buffer is AoS.
  void copyDevBufToHost_AoS(hipStream_t &stream, type_rho_dest_aos_desc dest_ptr, type_rho_buffer_aos_desc src_ptr,
                            const std::size_t src_offset, const std::size_t des_offset, std::size_t size);
  // similar as above, but the memory layout is SoA.
  void copyDevBufToHost_SoA(hipStream_t &stream, type_rho_dest_soa_desc dest_ptr, type_rho_buffer_soa_desc src_ptr,
                            const std::size_t src_offset, const std::size_t des_offset, std::size_t size);
};

#endif // MISA_MD_HIP_RHO_DOUBLE_BUFFER_IMP_H
