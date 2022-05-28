//
// Created by genshen on 2021/5/18.
//

#include <hip/hip_runtime.h>
#include <iostream>

#include "atom/atom_element.h"
#include "hip_macros.h" // from hip_pot lib

#include "kernels/hip_kernels.h"
#include "kernels/kernel_itl.hpp"
#include "md_hip_config.h"
#include "rho_double_buffer_imp.h"

RhoDoubleBufferImp::RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const db_buffer_data_desc data_desc,
                                       type_rho_src_desc src_atoms_desc, type_rho_dest_desc dest_atoms_desc,
                                       type_rho_buffer_desc _ptr_device_buf1, type_rho_buffer_desc _ptr_device_buf2,
                                       _hipDeviceDomain h_domain, const _hipDeviceNeiOffsets d_nei_offset,
                                       const double cutoff_radius)
    : DoubleBufferBaseImp(stream1, stream2, data_desc,
                          2 * h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x, 0,
                          h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x, src_atoms_desc,
                          dest_atoms_desc, _ptr_device_buf1, _ptr_device_buf2),
      h_domain(h_domain), d_nei_offset(d_nei_offset), cutoff_radius(cutoff_radius),
      atoms_per_layer(h_domain.ext_size_x * h_domain.ext_size_y) {

  constexpr int threads_per_block = 256;
  this->kernel_config_block_dim = dim3(threads_per_block);

  // One thread only process one atom.
  // note: size_x in h_domain is double
  const _type_atom_count local_atoms_num = h_domain.box_size_x * h_domain.box_size_y * h_domain.box_size_z;
  int blocks_num = (local_atoms_num - 1) / threads_per_block + 1;
  this->kernel_config_grid_dim = dim3(blocks_num);

  debug_printf("blocks: %d, threads: %d\n", blocks_num, threads_per_block);

}

void RhoDoubleBufferImp::calcAsync(hipStream_t &stream, const int block_id) {
  unsigned int data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  type_rho_buffer_desc d_p = (block_id % 2 == 0) ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  // atoms number to be calculated in this block
  const std::size_t atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  (itl_atoms_pair<tp_device_rho, ModeRho>)<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(
      d_p.atoms, nullptr, d_nei_offset, data_start_index, data_end_index, cutoff_radius);
#endif
  // todo: SoA
}

void RhoDoubleBufferImp::copyFromHostToDeviceBuf(hipStream_t &stream, type_rho_buffer_desc dest_ptr,
                                                 type_rho_src_desc src_ptr, const std::size_t src_offset,
                                                 std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyHostToDevBuf_AoS(stream, dest_ptr, src_ptr, src_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyHostToDevBuf_SoA(stream, dest_ptr, src_ptr, src_offset, size);
#endif
}

void RhoDoubleBufferImp::copyHostToDevBuf_AoS(hipStream_t &stream, type_rho_buffer_aos_desc dest_ptr,
                                              type_rho_src_aos_desc src_ptr, const std::size_t src_offset,
                                              std::size_t size) {
  HIP_CHECK(
      hipMemcpyAsync(dest_ptr.atoms, src_ptr.atoms, sizeof(_cuAtomElement) * size, hipMemcpyHostToDevice, stream));
}
void RhoDoubleBufferImp::copyHostToDevBuf_SoA(hipStream_t &stream, type_rho_buffer_soa_desc dest_ptr,
                                              type_rho_src_soa_desc src_ptr, const std::size_t src_offset,
                                              std::size_t size) {
  // copy types and x[3].
  HIP_CHECK(hipMemcpyAsync(dest_ptr.types, src_ptr.types, sizeof(_type_atom_type_enum) * size, hipMemcpyHostToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.x, src_ptr.x, sizeof(_type_atom_location[HIP_DIMENSION]) * size,
                           hipMemcpyHostToDevice, stream));
  // memory set force
  HIP_CHECK(hipMemsetAsync(dest_ptr.rho, 0, sizeof(_type_atom_rho) * size, stream));
}

void RhoDoubleBufferImp::copyFromDeviceBufToHost(hipStream_t &stream, type_rho_dest_desc dest_ptr,
                                                 type_rho_buffer_desc src_ptr, const std::size_t src_offset,
                                                 const std::size_t des_offset, std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyDevBufToHost_AoS(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyDevBufToHost_SoA(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
}

void RhoDoubleBufferImp::copyDevBufToHost_AoS(hipStream_t &stream, type_rho_dest_aos_desc dest_ptr,
                                              type_rho_buffer_aos_desc src_ptr, const std::size_t src_offset,
                                              const std::size_t des_offset, std::size_t size) {
  HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
                           hipMemcpyDeviceToHost, stream));
}
void RhoDoubleBufferImp::copyDevBufToHost_SoA(hipStream_t &stream, type_rho_dest_soa_desc dest_ptr,
                                              type_rho_buffer_soa_desc src_ptr, const std::size_t src_offset,
                                              const std::size_t des_offset, std::size_t size) {
  // copy rho back
  HIP_CHECK(hipMemcpyAsync(dest_ptr.rho + des_offset, src_ptr.rho + src_offset, sizeof(_type_atom_rho) * size,
                           hipMemcpyDeviceToHost, stream));
}