//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H
#define MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H

#include <hip/hip_runtime.h>

#include "../global_ops.h"
#include "atom/atom_element.h"
#include "double_buffer.h"
#include "double_buffer_base_imp.hpp"
#include "kernels/hip_kernels.h"
#include "memory/device_atoms.h"

typedef device_atoms::_type_buffer_desc type_f_buffer_desc;
typedef device_atoms::_type_atom_list_desc type_f_src_desc;
typedef device_atoms::_type_atom_list_desc type_f_dest_desc;

typedef device_atoms::_type_dev_buffer_aos type_f_buffer_aos_desc;
typedef device_atoms::_type_dev_buffer_soa type_f_buffer_soa_desc;

typedef device_atoms::_type_atom_list_soa type_f_src_soa_desc;
typedef device_atoms::_type_atom_list_aos type_f_src_aos_desc;
typedef device_atoms::_type_atom_list_soa type_f_dest_soa_desc;
typedef device_atoms::_type_atom_list_aos type_f_dest_aos_desc;

class ForceDoubleBufferImp : public DoubleBufferBaseImp<type_f_buffer_desc, type_f_src_desc, type_f_dest_desc> {
public:
  /**
   * double buffer implementation for force calculation.
   * @param stream1 Hip stream for buffer 1
   * @param stream2  Hip stream for buffer 2
   * @param data_desc source data descriptor as an input for calculation.
   * @param src_atoms_desc source data descriptor where it copies from when coping data from host side to double buffer.
   *   Under AoS memory layout, it can be the lattice atoms array in current MPI process (including ghost regions).
   * @param dest_atoms_desc destination data descriptor where it fetches to when fetching data from device side double
   * buffer to host side.
   *  Under AoS memory layout, it may keep the same as @param src_atoms_desc.
   * @param _ptr_device_buf1 buffer 1
   * @param _ptr_device_buf2 buffer 2
   * @param h_domain simulation domain
   * @param d_nei_offset the neighbor offset array in our MD for searching neighbor atoms.
   * @param cutoff_radius the cutoff radius.
   */
  ForceDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const db_buffer_data_desc data_desc,
                       type_f_src_desc src_atoms_desc, type_f_dest_desc dest_atoms_desc,
                       type_f_buffer_desc _ptr_device_buf1, type_f_buffer_desc _ptr_device_buf2,
                       _hipDeviceDomain h_domain, const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius);

  void calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) override;

private:
  const _hipDeviceDomain h_domain;
  const _hipDeviceNeiOffsets d_nei_offset; // fixme: remove it
  const double cutoff_radius;
  const _type_atom_count atoms_per_layer; // atoms in each layer at z dimension.
  dim3 kernel_config_block_dim;
  dim3 kernel_config_grid_dim;

private:
  void copyFromHostToDeviceBuf(hipStream_t &stream, type_f_buffer_desc dest_ptr, type_f_src_desc src_ptr,
                               const std::size_t src_offset, std::size_t size) override;
  void copyFromDeviceBufToHost(hipStream_t &stream, type_f_dest_desc dest_ptr, type_f_buffer_desc src_ptr,
                               const std::size_t src_offset, const std::size_t des_offset, std::size_t size) override;

private:
  /**
   * Launch the kernel to calculate force if the memory layout is Array of Struct mode.
   * @param stream Hip Stream
   * @param d_p double buffer descriptor
   * @param atom_num_calc the number of atoms to be calculated in current data block.
   * @param data_start_index block item start index.
   * @param data_end_index block item ending index.
   */
  void launchKernelMemLayoutAoS(hipStream_t &stream, type_f_buffer_aos_desc d_p, const _type_atom_count atom_num_calc,
                                const DoubleBuffer::tp_block_item_idx data_start_index,
                                const DoubleBuffer::tp_block_item_idx data_end_index);

  /**
   * Launch the kernel to calculate force if the memory layout is Struct of Array mode.
   * @param stream Hip Stream
   * @param d_p double buffer descriptor
   * @param atom_num_calc the number of atoms to be calculated in current data block.
   * @param data_start_index block item start index.
   * @param data_end_index block item ending index.
   */
  void launchKernelMemLayoutSoA(hipStream_t &stream, type_f_buffer_soa_desc d_p, const _type_atom_count atom_num_calc,
                                const DoubleBuffer::tp_block_item_idx data_start_index,
                                const DoubleBuffer::tp_block_item_idx data_end_index);

private:
  // copy atoms data from host side to device buffer, where the memory layout of the host atoms and buffer is AoS.
  void copyHostToDevBuf_AoS(hipStream_t &stream, type_f_buffer_aos_desc dest_ptr, type_f_src_aos_desc src_ptr,
                            const std::size_t src_offset, std::size_t size);
  // similar as above, but the memory layout is SoA.
  void copyHostToDevBuf_SoA(hipStream_t &stream, type_f_buffer_soa_desc dest_ptr, type_f_src_soa_desc src_ptr,
                            const std::size_t src_offset, std::size_t size);

  // copy atoms data from device buffer side to host side, where the memory layout of the host atoms and buffer is AoS.
  void copyDevBufToHost_AoS(hipStream_t &stream, type_f_dest_aos_desc dest_ptr, type_f_buffer_aos_desc src_ptr,
                            const std::size_t src_offset, const std::size_t des_offset, std::size_t size);
  // similar as above, but the memory layout is SoA.
  void copyDevBufToHost_SoA(hipStream_t &stream, type_f_dest_soa_desc dest_ptr, type_f_buffer_soa_desc src_ptr,
                            const std::size_t src_offset, const std::size_t des_offset, std::size_t size);
};

#endif // MISA_MD_HIP_FORCE_DOUBLE_BUFFER_IMP_H
