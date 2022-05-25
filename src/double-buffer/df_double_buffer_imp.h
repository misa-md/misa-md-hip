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
#include "kernels/hip_kernels.h"
#include "memory/device_atoms.h"

typedef device_atoms::_type_buffer_desc type_df_buffer_desc;
typedef _type_atom_list_collection type_df_src_desc;
typedef _type_atom_list_collection type_df_dest_desc;
typedef tp_device_df type_df_fetch_desc;

/**
 * double buffer implementation for calculating derivative of embedded energy: df.
 */
class DfDoubleBufferImp
    : public DoubleBufferBaseImp<type_df_buffer_desc, type_df_src_desc, type_df_dest_desc, type_df_fetch_desc> {
public:
  /**
   * All parameters are the same as rho calculation.
   * @param stream1,stream2 2 hip streams used for syncing buffer 1 and buffer 2 (e.g. data copying).
   * @param blocks total blocks for double buffer.
   * @param data_len total z-layers of atoms on current processor (not include ghost regions).
   * @param _ptr_atoms atoms data descriptor in host side.
   * @param _ptr_device_buf1, _ptr_device_buf2 2 buffer memory descriptor on device side
   * @param _d_dfs the results data descriptor on device side (calculating results will writ to this array).
   * @param h_domain domain information
   */
  DfDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len,
                    type_df_src_desc _ptr_atoms, type_df_buffer_desc _ptr_device_buf1,
                    type_df_buffer_desc _ptr_device_buf2, type_df_fetch_desc *_d_dfs, _hipDeviceDomain h_domain);

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const int block_id) override;

private:
  // lattice atoms array in current MPI process (including ghost regions)
  type_df_src_desc ptr_atoms;
  type_df_fetch_desc *d_dfs = nullptr;
  const _hipDeviceDomain h_domain;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;

private:
  void copyFromHostToDeviceBuf(hipStream_t &stream, type_df_buffer_desc dest_ptr, type_df_src_desc src_ptr,
                               const std::size_t src_offset, std::size_t size) override;
  void copyFromDeviceBufToHost(hipStream_t &stream, type_df_dest_desc dest_ptr, type_df_buffer_desc src_ptr,
                               const std::size_t src_offset, const std::size_t des_offset, std::size_t size) override;
};

#endif // MISA_MD_HIP_DF_DOUBLE_BUFFER_IMP_H
