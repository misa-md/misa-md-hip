//
// Created by genshen on 2021/5/18.
//

#include <hip/hip_runtime.h>
#include <iostream>

#include "atom/atom_element.h"
#include "hip_pot_macros.h" // from hip_pot lib

#include "../kernels/soa_thread_atom.h"
#include "kernels/hip_kernels.h"
#include "kernels/kernel_itl.hpp"
#include "kernels/soa_eam_pair.hpp"
#include "kernels/soa_wf_atom.h"
#include "md_hip_building_config.h"
#include "md_hip_config.h"
#include "optimization_level.h"
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
      atoms_per_layer(h_domain.box_size_x * h_domain.box_size_y) {
  // note: size_x in h_domain is double
}

void RhoDoubleBufferImp::calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) {
  DoubleBuffer::tp_block_item_idx data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  type_rho_buffer_desc d_p = (block_id % 2 == 0) ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  // atoms number to be calculated in this data-block.
  // note: size_x in variable atoms_per_layer is double.
  const _type_atom_count atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  launchKernelMemLayoutAoS(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  launchKernelMemLayoutSoA(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
}

void RhoDoubleBufferImp::launchKernelMemLayoutAoS(hipStream_t &stream, type_rho_buffer_aos_desc d_p,
                                                  const _type_atom_count atom_num_calc,
                                                  const DoubleBuffer::tp_block_item_idx data_start_index,
                                                  const DoubleBuffer::tp_block_item_idx data_end_index) {
  constexpr int threads_per_block = 256;
  this->kernel_config_block_dim = dim3(threads_per_block);

  // One thread only process one atom.
  int blocks_num = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
  this->kernel_config_grid_dim = dim3(blocks_num);

  debug_printf("blocks: %d, threads: %d\n", blocks_num, threads_per_block);

  (itl_atoms_pair<tp_device_rho, ModeRho>)<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(
      d_p.atoms, nullptr, d_nei_offset, data_start_index, data_end_index, cutoff_radius);
}

void RhoDoubleBufferImp::launchKernelMemLayoutSoA(hipStream_t &stream, type_rho_buffer_soa_desc d_p,
                                                  const _type_atom_count atom_num_calc,
                                                  const DoubleBuffer::tp_block_item_idx data_start_index,
                                                  const DoubleBuffer::tp_block_item_idx data_end_index) {
  if (KERNEL_STRATEGY == KERNEL_STRATEGY_THREAD_ATOM) {
    constexpr int threads_per_block = 256;
    int grid_dim = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
    (md_nei_itl_soa<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec1, double,
                    _type_d_vec1>)<<<grid_dim, threads_per_block, 0, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec1 *>(d_p.rho),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius);
  } else {
    constexpr int threads_per_block = 256;
    constexpr int wf_size_per_block = threads_per_block / __WAVE_SIZE__;
    int grid_dim = atom_num_calc / wf_size_per_block + (atom_num_calc % wf_size_per_block == 0 ? 0 : 1);
    (md_nei_itl_wf_atom_soa<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec1, double,
                            _type_d_vec1>)<<<grid_dim, threads_per_block, 0, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec1 *>(d_p.rho),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius);
  }
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
  HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
                           hipMemcpyHostToDevice, stream));
}

void RhoDoubleBufferImp::copyHostToDevBuf_SoA(hipStream_t &stream, type_rho_buffer_soa_desc dest_ptr,
                                              type_rho_src_soa_desc src_ptr, const std::size_t src_offset,
                                              std::size_t size) {
  // copy types and x[3].
  HIP_CHECK(hipMemcpyAsync(dest_ptr.types, src_ptr.types + src_offset, sizeof(_type_atom_type_enum) * size,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.x, src_ptr.x + src_offset, sizeof(_type_atom_location[HIP_DIMENSION]) * size,
                           hipMemcpyHostToDevice, stream));
  if (global_config::use_newtons_third_law()) {
    // memory set rho.
    // maybe memset is unnecessary if newton's third law is disabled, because it store into memory directly under this
    // condition.
    HIP_CHECK(hipMemsetAsync(dest_ptr.rho, 0, sizeof(_type_atom_rho) * size, stream));
  }
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
  if (!global_config::use_newtons_third_law()) {
    // if newton's third law is not enabled, we need also to copy df.
    // because the full df is calculated in rho calculation.
    HIP_CHECK(hipMemcpyAsync(dest_ptr.df + des_offset, src_ptr.df + src_offset, sizeof(_type_atom_df) * size,
                             hipMemcpyDeviceToHost, stream));
  }
}
