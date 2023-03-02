//
// Created by genshen on 2021/5/23.
//

#include <hip/hip_runtime.h>
#include <iostream>

#include "atom/atom_element.h"
#include "hip_pot_macros.h" // from hip_pot lib

#include "force_double_buffer_imp.h"
#include "kernels/aos-thread-atom/kernel_itl.hpp"
#include "kernels/hip_kernels.h"
#include "kernels/soa-block-atom/soa_block_atom.hpp"
#include "kernels/soa-thread-atom/soa_thread_atom.h"
#include "kernels/soa-wf-atom/soa_wf_atom.h"
#include "kernels/soa_eam_pair.hpp"
#include "kernels/types/vec3.hpp"
#include "md_hip_building_config.h"
#include "md_hip_config.h"
#include "optimization_level.h"

ForceDoubleBufferImp::ForceDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2,
                                           const db_buffer_data_desc data_desc, type_f_src_desc src_atoms_desc,
                                           type_f_dest_desc dest_atoms_desc, type_f_buffer_desc _ptr_device_buf1,
                                           type_f_buffer_desc _ptr_device_buf2, _hipDeviceDomain h_domain,
                                           const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius)
    : DoubleBufferBaseImp(stream1, stream2, data_desc, db_buffer_data_copy_option::build_copy_option(h_domain),
                          src_atoms_desc, dest_atoms_desc, _ptr_device_buf1, _ptr_device_buf2),
      h_domain(h_domain), d_nei_offset(d_nei_offset), cutoff_radius(cutoff_radius),
      atoms_per_layer(h_domain.box_size_x * h_domain.box_size_y) {

  // note: size_x in h_domain is double
}

void ForceDoubleBufferImp::calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) {
  DoubleBuffer::tp_block_item_idx data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  type_f_buffer_desc d_p = (block_id % 2 == 0) ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  // atoms number to be calculated in this block
  const std::size_t atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  // One thread only process one atom.
  launchKernelMemLayoutAoS(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  launchKernelMemLayoutSoA(stream, d_p, atom_num_calc, data_start_index, data_end_index);
#endif
}

void ForceDoubleBufferImp::launchKernelMemLayoutAoS(hipStream_t &stream, type_f_buffer_aos_desc d_p,
                                                    const _type_atom_count atom_num_calc,
                                                    const DoubleBuffer::tp_block_item_idx data_start_index,
                                                    const DoubleBuffer::tp_block_item_idx data_end_index) {
  constexpr int threads_per_block = 256;
  this->kernel_config_block_dim = dim3(threads_per_block);
  int blocks_num = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
  this->kernel_config_grid_dim = dim3(blocks_num);

  (itl_atoms_pair<tp_device_force,
                  ModeForce>)<<<dim3(kernel_config_grid_dim), dim3(kernel_config_block_dim), 0, stream>>>(
      d_p.atoms, nullptr, d_nei_offset, data_start_index, data_end_index, cutoff_radius);
}

void ForceDoubleBufferImp::launchKernelMemLayoutSoA(hipStream_t &stream, type_f_buffer_soa_desc d_p,
                                                    const _type_atom_count atom_num_calc,
                                                    const DoubleBuffer::tp_block_item_idx data_start_index,
                                                    const DoubleBuffer::tp_block_item_idx data_end_index) {
  if (KERNEL_STRATEGY == KERNEL_STRATEGY_THREAD_ATOM) {
    constexpr int threads_per_block = 256;
    int grid_dim = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);

    (md_nei_itl_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
                    _type_d_vec3>)<<<grid_dim, threads_per_block, 0, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec3 *>(d_p.f),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius);
  } else if (KERNEL_STRATEGY == KERNEL_STRATEGY_WF_ATOM) {
    constexpr int threads_per_block = 256;
    constexpr int wf_size_per_block = threads_per_block / __WAVE_SIZE__;
    int grid_dim = atom_num_calc / wf_size_per_block + (atom_num_calc % wf_size_per_block == 0 ? 0 : 1);

    (md_nei_itl_wf_atom_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
                            _type_d_vec3>)<<<grid_dim, threads_per_block, 0, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec3 *>(d_p.f),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius);
  } else {
    constexpr int threads_per_block = 128;
    int grid_dim = atom_num_calc;
    const int shared_size =
        soa_block_atom_kernel_shared_size<double, _type_d_vec3, _type_atom_type_kernel, _type_atom_index_kernel>(
            d_nei_offset, threads_per_block / __WAVE_SIZE__);

    (md_nei_itl_block_atom_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3,
                               double, _type_d_vec3,
                               threads_per_block>)<<<grid_dim, threads_per_block, shared_size, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec3 *>(d_p.f),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius);
  }
}

void ForceDoubleBufferImp::copyFromHostToDeviceBuf(hipStream_t &stream, type_f_buffer_desc dest_ptr,
                                                   type_f_src_desc src_ptr, const std::size_t src_offset,
                                                   std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyHostToDevBuf_AoS(stream, dest_ptr, src_ptr, src_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyHostToDevBuf_SoA(stream, dest_ptr, src_ptr, src_offset, size);
#endif
}

void ForceDoubleBufferImp::copyHostToDevBuf_AoS(hipStream_t &stream, type_f_buffer_aos_desc dest_ptr,
                                                type_f_src_aos_desc src_ptr, const std::size_t src_offset,
                                                std::size_t size) {
  HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
                           hipMemcpyHostToDevice, stream));
}

void ForceDoubleBufferImp::copyHostToDevBuf_SoA(hipStream_t &stream, type_f_buffer_soa_desc dest_ptr,
                                                type_f_src_soa_desc src_ptr, const std::size_t src_offset,
                                                std::size_t size) {
  // only copy type, x[3], rho, and df.
  HIP_CHECK(hipMemcpyAsync(dest_ptr.types, src_ptr.types + src_offset, sizeof(_type_atom_type_enum) * size,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.x, src_ptr.x + src_offset, sizeof(_type_atom_location[HIP_DIMENSION]) * size,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.rho, src_ptr.rho + src_offset, sizeof(_type_atom_rho) * size, hipMemcpyHostToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(dest_ptr.df, src_ptr.df + src_offset, sizeof(_type_atom_df) * size, hipMemcpyHostToDevice,
                           stream));
  // memory set force
  HIP_CHECK(hipMemsetAsync(dest_ptr.f, 0, sizeof(_type_atom_force[HIP_DIMENSION]) * size, stream));
}

void ForceDoubleBufferImp::copyFromDeviceBufToHost(hipStream_t &stream, type_f_dest_desc dest_ptr,
                                                   type_f_buffer_desc src_ptr, const std::size_t src_offset,
                                                   const std::size_t des_offset, std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  copyDevBufToHost_AoS(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyDevBufToHost_SoA(stream, dest_ptr, src_ptr, src_offset, des_offset, size);
#endif
}
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
void ForceDoubleBufferImp::copyDevBufToHost_AoS(hipStream_t &stream, type_f_dest_aos_desc dest_ptr,
                                                type_f_buffer_aos_desc src_ptr, const std::size_t src_offset,
                                                const std::size_t des_offset, std::size_t size) {
  if (one_process_multi_gpus_flag && global_config::use_newtons_third_law()) {
    int device_id;
    hipGetDevice(&device_id);

    int size_d2d = h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x;

    _cuAtomElement *cal_buffer;
    HIP_CHECK(hipMalloc((void **)&cal_buffer, sizeof(_cuAtomElement) * size_d2d));
    constexpr int threads_per_block = 256;
    int grid_dim = size_d2d / threads_per_block + (size_d2d % threads_per_block == 0 ? 0 : 1);
    
    if (device_id == 0) {
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[1].atoms, sizeof(_cuAtomElement) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      
      vector_add_aos_force<<<grid_dim, threads_per_block, 0, stream>>>(
                          reinterpret_cast<_cuAtomElement *>(device_atoms::d_atoms_buffer1[0].atoms + size - 2 * size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset, device_atoms::d_atoms_buffer1[0].atoms, sizeof(_cuAtomElement) * (size - size_d2d),
                hipMemcpyDeviceToHost, stream));
    } else if (device_id == gpu_num_per_node - 1) {
      _type_lattice_size max_size_z_per_gpu = (h_domain.box_size_z - 1) / gpu_num_per_node + 1;
      int max_size = (max_size_z_per_gpu + 2 * h_domain.ghost_size_z) * h_domain.ext_size_y * h_domain.ext_size_x;
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id - 1].atoms + max_size - size_d2d, sizeof(_cuAtomElement) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_aos_force<<<grid_dim, threads_per_block, 0, stream>>>(
                           reinterpret_cast<_cuAtomElement *>(device_atoms::d_atoms_buffer1[device_id].atoms + size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset + size_d2d, device_atoms::d_atoms_buffer1[device_id].atoms + size_d2d, sizeof(_cuAtomElement) * (size - size_d2d),
                hipMemcpyDeviceToHost, stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id + 1].atoms, sizeof(_cuAtomElement) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_aos_rho<<<grid_dim, threads_per_block, 0, stream>>>(
                           reinterpret_cast<_cuAtomElement *>(device_atoms::d_atoms_buffer1[device_id].atoms + size - 2 * size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id - 1].atoms + size - size_d2d, sizeof(_cuAtomElement) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_aos_force<<<grid_dim, threads_per_block, 0, stream>>>(
                            reinterpret_cast<_cuAtomElement *>(device_atoms::d_atoms_buffer1[device_id].atoms + size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset + size_d2d, device_atoms::d_atoms_buffer1[device_id].atoms + size_d2d, sizeof(_cuAtomElement) * (size - 2 * size_d2d),
                hipMemcpyDeviceToHost, stream));
    }
  } else {
    HIP_CHECK(hipMemcpyAsync(dest_ptr.atoms + des_offset, src_ptr.atoms + src_offset, sizeof(_cuAtomElement) * size,
                           hipMemcpyDeviceToHost, stream));
  }
}
#endif

#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
void ForceDoubleBufferImp::copyDevBufToHost_SoA(hipStream_t &stream, type_f_dest_soa_desc dest_ptr,
                                                type_f_buffer_soa_desc src_ptr, const std::size_t src_offset,
                                                const std::size_t des_offset, std::size_t size) {
  // only copy force[3] back.
  if (one_process_multi_gpus_flag && global_config::use_newtons_third_law()) {
    int device_id;
    hipGetDevice(&device_id);
    int size_d2d = h_domain.ghost_size_z * h_domain.ext_size_y * h_domain.ext_size_x;
    
    _type_atom_force (*cal_buffer)[HIP_DIMENSION];
    HIP_CHECK(hipMalloc((void **)&cal_buffer, sizeof(_type_atom_force[HIP_DIMENSION]) * size_d2d));
    constexpr int threads_per_block = 256;
    int grid_dim = size_d2d / threads_per_block + (size_d2d % threads_per_block == 0 ? 0 : 1);
    
    if (device_id == 0) {
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[1].f, sizeof(_type_atom_force[HIP_DIMENSION]) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_soa_force<<<grid_dim, threads_per_block, 0, stream>>>(
                          reinterpret_cast<double (*)[HIP_DIMENSION]>(device_atoms::d_atoms_buffer1[0].f + size - 2 * size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.f + des_offset, device_atoms::d_atoms_buffer1[0].f, sizeof(_type_atom_force[HIP_DIMENSION]) * (size - size_d2d),
                hipMemcpyDeviceToHost, stream));
    } else if (device_id == gpu_num_per_node - 1) {
      _type_lattice_size max_size_z_per_gpu = (h_domain.box_size_z - 1) / gpu_num_per_node + 1;
      int max_size = (max_size_z_per_gpu + 2 * h_domain.ghost_size_z) * h_domain.ext_size_y * h_domain.ext_size_x;
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id - 1].f + max_size - size_d2d, sizeof(_type_atom_force[HIP_DIMENSION]) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_soa_force<<<grid_dim, threads_per_block, 0, stream>>>(
                           reinterpret_cast<double (*)[HIP_DIMENSION]>(device_atoms::d_atoms_buffer1[device_id].f + size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.f + des_offset + size_d2d, device_atoms::d_atoms_buffer1[device_id].f + size_d2d, sizeof(_type_atom_force[HIP_DIMENSION]) * (size - size_d2d),
                hipMemcpyDeviceToHost, stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id + 1].f, sizeof(_type_atom_force[HIP_DIMENSION]) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_soa_force<<<grid_dim, threads_per_block, 0, stream>>>(
                           reinterpret_cast<double (*)[HIP_DIMENSION]>(device_atoms::d_atoms_buffer1[device_id].f + size - 2 * size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(cal_buffer, device_atoms::d_atoms_buffer1[device_id - 1].f + size - size_d2d, sizeof(_type_atom_force[HIP_DIMENSION]) * size_d2d,
                hipMemcpyDeviceToDevice, stream));
      vector_add_soa_force<<<grid_dim, threads_per_block, 0, stream>>>(
                            reinterpret_cast<double (*)[HIP_DIMENSION]>(device_atoms::d_atoms_buffer1[device_id].f + size_d2d), cal_buffer, size_d2d);
      HIP_CHECK(hipMemcpyAsync(dest_ptr.f + des_offset + size_d2d, device_atoms::d_atoms_buffer1[device_id].f + size_d2d, sizeof(_type_atom_force[HIP_DIMENSION]) * (size - 2 * size_d2d),
                hipMemcpyDeviceToHost, stream));
    }
  } else {
    HIP_CHECK(hipMemcpyAsync(dest_ptr.f + des_offset, src_ptr.f + src_offset,
                           sizeof(_type_atom_force[HIP_DIMENSION]) * size, hipMemcpyDeviceToHost, stream));
  }
}
#endif
