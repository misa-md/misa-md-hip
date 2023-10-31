//
// Created by lihuizhao on 2023/10/17.
//

#include <hip/hip_runtime.h>
#include <iostream>

#include "atom/atom_element.h"
#include "hip_pot_macros.h" // from hip_pot lib

#include "bitmap_double_buffer_imp.h"
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

#include "kernels/common/bitmap.hpp"


BitmapDoubleBufferImp::BitmapDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2,
                                           const db_buffer_data_desc data_desc, type_f_src_desc src_atoms_desc,
                                           type_f_dest_desc dest_atoms_desc, type_f_buffer_desc _ptr_device_buf1,
                                           type_f_buffer_desc _ptr_device_buf2, _hipDeviceDomain h_domain,
                                           const _hipDeviceNeiOffsets d_nei_offset, const double cutoff_radius,int* bitmap_mem)
    : DoubleBufferBaseImp(stream1, stream2, data_desc, db_buffer_data_copy_option::build_copy_option(h_domain),
                          src_atoms_desc, dest_atoms_desc, _ptr_device_buf1, _ptr_device_buf2),
      h_domain(h_domain), d_nei_offset(d_nei_offset), cutoff_radius(cutoff_radius),bitmap_mem(bitmap_mem),
      atoms_per_layer(h_domain.box_size_x * h_domain.box_size_y) {

  // note: size_x in h_domain is double
}

void BitmapDoubleBufferImp::schedule() {
  // fill data to buffer 1
  fillBufferWrapper(0); // copy with async
  for (tp_data_block_id i = 0; i < blocks; i++) {
    waitCalc(i - 1);           // wait calculation finishing of buffer 1 if i is odd, otherwise, buffer 2.
    fillBufferWrapper(i + 1);  // fill next block data to buffer 1 if i is odd, otherwise, buffer 2.
    // wait buffer 2 coping finishing (It can also wait another buffer's calculation),
    // if i is odd, otherwise, buffer 1.
    waitComm(i);
    calcAsync(i % 2 == 0 ? stream1 : stream2, i); // calculate buffer 2 if i is odd, otherwise, buffer 1.
  }
  // fetch data from buffer 2 or buffer 1, if necessary
  waitCalc(blocks - 1);
}

void BitmapDoubleBufferImp::calcAsync(hipStream_t &stream, const DoubleBuffer::tp_data_block_id block_id) {
  DoubleBuffer::tp_block_item_idx data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);

  type_f_buffer_desc d_p = (block_id % 2 == 0) ? d_ptr_device_buf1 : d_ptr_device_buf2; // ghost is included in d_p
  // atoms number to be calculated in this block
  const std::size_t atom_num_calc = atoms_per_layer * (data_end_index - data_start_index);
  launchKernelMemLayoutSoA(stream, d_p, atom_num_calc, data_start_index, data_end_index);
}



void BitmapDoubleBufferImp::launchKernelMemLayoutSoA(hipStream_t &stream, type_f_buffer_soa_desc d_p,
                                                    const _type_atom_count atom_num_calc,
                                                    const DoubleBuffer::tp_block_item_idx data_start_index,
                                                    const DoubleBuffer::tp_block_item_idx data_end_index) {
    constexpr int threads_per_block = 256;
    int grid_dim = atom_num_calc / threads_per_block + (atom_num_calc % threads_per_block == 0 ? 0 : 1);
    //printf("\natom_num_calc=%lu\n",atom_num_calc);

    (md_nei_bitmap<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
                    _type_d_vec3>)<<<grid_dim, threads_per_block, 0, stream>>>(
        d_p.x, reinterpret_cast<_type_atom_type_kernel *>(d_p.types), d_p.df, reinterpret_cast<_type_d_vec3 *>(d_p.f),
        atom_num_calc, d_nei_offset, h_domain, cutoff_radius,bitmap_mem);

}

void BitmapDoubleBufferImp::copyFromHostToDeviceBuf(hipStream_t &stream, type_f_buffer_desc dest_ptr,
                                                   type_f_src_desc src_ptr, const std::size_t src_offset,
                                                   std::size_t size) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  copyHostToDevBuf_SoA(stream, dest_ptr, src_ptr, src_offset, size);
#endif
}



void BitmapDoubleBufferImp::copyHostToDevBuf_SoA(hipStream_t &stream, type_f_buffer_soa_desc dest_ptr,
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
