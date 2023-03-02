//
// Created by genshen on 2021/5/21.
//

#include "df_double_buffer_imp.h"
#include "hip_pot_macros.h" // from hip_pot lib

DfDoubleBufferImp::DfDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const db_buffer_data_desc data_desc,
                                     type_df_src_desc src_atoms_desc, type_df_dest_desc dest_atoms_desc,
                                     type_df_buffer_desc _ptr_device_buf1, type_df_buffer_desc _ptr_device_buf2,
                                     _hipDeviceDomain h_domain)
    : DoubleBufferBaseImp(stream1, stream2, data_desc, db_buffer_data_copy_option::build_copy_option(h_domain),
                          src_atoms_desc, dest_atoms_desc, _ptr_device_buf1, _ptr_device_buf2),
      h_domain(h_domain), atoms_per_layer(h_domain.box_size_x * h_domain.box_size_y) {
  // note: size_x in h_domain is double
}

void DfDoubleBufferImp::calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) {
  DoubleBuffer::tp_block_item_idx data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  type_df_buffer_desc d_p = block_id % 2 == 0 ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  // atoms number to be calculated in this data block
  const std::size_t atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);

#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  // One thread only process one atom.
  launchKernelMemLayoutAoS(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  launchKernelMemLayoutSoA(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
}

void DfDoubleBufferImp::launchKernelMemLayoutAoS(hipStream_t &stream, type_df_buffer_aos_desc d_p,
                                                 const _type_atom_count atom_num_calc,
                                                 const DoubleBuffer::tp_block_item_idx data_start_index,
                                                 const DoubleBuffer::tp_block_item_idx data_end_index) {
  constexpr int threads_per_block = 256;
  int blocks_num = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
  this->kernel_config_block_dim = dim3(threads_per_block);
  this->kernel_config_grid_dim = dim3(blocks_num);

  cal_df_aos<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(d_p.atoms, data_start_index,
                                                                                         data_end_index);
}

void DfDoubleBufferImp::launchKernelMemLayoutSoA(hipStream_t &stream, type_df_buffer_soa_desc d_p,
                                                 const _type_atom_count atom_num_calc,
                                                 const DoubleBuffer::tp_block_item_idx data_start_index,
                                                 const DoubleBuffer::tp_block_item_idx data_end_index) {
  constexpr int threads_per_block = 256;
  int blocks_num = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
  this->kernel_config_block_dim = dim3(threads_per_block);
  this->kernel_config_grid_dim = dim3(blocks_num);

  (cal_df_soa<_type_atom_type_enum, _type_atom_rho,
              _type_atom_count>)<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(
      d_p.rho, d_p.df, d_p.types, atom_num_calc, h_domain);
}

void DfDoubleBufferImp::copyFromHostToDeviceBuf(hipStream_t &stream, type_df_buffer_desc dest_ptr,
                                                type_df_src_desc src_ptr, const std::size_t src_offset,
                                                std::size_t size) {
  // todo: use offset:
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyHostToDevBuf_AoS(stream, dest_ptr, src_ptr, src_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyHostToDevBuf_SoA(stream, dest_ptr, src_ptr, src_offset, size);
#endif
}

void DfDoubleBufferImp::copyHostToDevBuf_AoS(hipStream_t &stream, type_df_buffer_aos_desc dest_ptr,
                                             type_df_src_aos_desc src_ptr, const std::size_t src_offset,
                                             std::size_t size) {
  HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
                           hipMemcpyHostToDevice, stream));
}

void DfDoubleBufferImp::copyHostToDevBuf_SoA(hipStream_t &stream, type_df_buffer_soa_desc dest_ptr,
                                             type_df_src_soa_desc src_ptr, const std::size_t src_offset,
                                             std::size_t size) {
  HIP_CHECK(hipMemcpyAsync(dest_ptr.types, src_ptr.types + src_offset, sizeof(_type_atom_type_enum) * size,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.x, src_ptr.x + src_offset, sizeof(_type_atom_location[HIP_DIMENSION]) * size,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.rho, src_ptr.rho + src_offset, sizeof(_type_atom_rho) * size, hipMemcpyHostToDevice,
                           stream));
  // memory set:
  HIP_CHECK(hipMemsetAsync(dest_ptr.df, 0, sizeof(_type_atom_rho) * size, stream));
}

void DfDoubleBufferImp::copyFromDeviceBufToHost(hipStream_t &stream, type_df_dest_desc dest_ptr,
                                                type_df_buffer_desc src_ptr, const std::size_t src_offset,
                                                const std::size_t des_offset, std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyDevBufToHost_AoS(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyDevBufToHost_SoA(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
}

void DfDoubleBufferImp::copyDevBufToHost_AoS(hipStream_t &stream, type_df_dest_aos_desc dest_ptr,
                                             type_df_buffer_aos_desc src_ptr, const std::size_t src_offset,
                                             const std::size_t des_offset, std::size_t size) {
  if (one_process_multi_gpus_flag && global_config::use_newtons_third_law()) {
    std::size_t size_d2d = h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x;
    HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset + size_d2d, src_ptr.atoms + src_offset + size_d2d, sizeof(_cuAtomElement) * (size - 2 * size_d2d),
              hipMemcpyDeviceToHost, stream));
  } else {
    HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
              hipMemcpyDeviceToHost, stream));
  }
}

void DfDoubleBufferImp::copyDevBufToHost_SoA(hipStream_t &stream, type_df_dest_soa_desc dest_ptr,
                                             type_df_buffer_soa_desc src_ptr, const std::size_t src_offset,
                                             const std::size_t des_offset, std::size_t size) {
  if (one_process_multi_gpus_flag && global_config::use_newtons_third_law()) {
    std::size_t size_d2d = h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x;
    HIP_CHECK(hipMemcpyAsync(dest_ptr.df + des_offset + size_d2d, src_ptr.df + src_offset + size_d2d, sizeof(_type_atom_df) * (size - 2 * size_d2d),
              hipMemcpyDeviceToHost, stream));
  } else {
    HIP_CHECK(hipMemcpyAsync(dest_ptr.df + des_offset, src_ptr.df + src_offset, sizeof(_type_atom_df) * size,
                           hipMemcpyDeviceToHost, stream));
  }
}
