#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>

#include "arch/arch_imp.h"
#include "hip_macros.h" // from hip_pot lib
#include "hip_pot.h"
#include "kernels/hip_kernels.h"

#include "cli.h"
#include "double-buffer/df_double_buffer_imp.h"
#include "double-buffer/force_double_buffer_imp.h"
#include "double-buffer/rho_double_buffer_imp.h"
#include "global_ops.h"
#include "kernel_wrapper.h"
#include "md_hip_config.h"
#include "memory/device_atoms.h"
#include "optimization_level.h"

//定义线程块各维线程数
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 16

_hipDeviceDomain h_domain;
// double *d_constValue_double;
_hipDeviceNeiOffsets d_nei_offset;

void hip_env_init() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  HIP_CHECK(hipSetDevice(0));
  //设备信息，可以打印一些资源值
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;
  std::cout << "hip Device prop succeeded " << std::endl;
  std::cout << "batches number: " << batches_cli << std::endl;
}

void hip_env_clean() {
  // fixme hip_pot::destroyDevicePotTables(d_pot);
  // fixme HIP_CHECK(hipFree(d_atoms));
}

void hip_domain_init(const comm::BccDomain *p_domain) {
  const int lolocalx = p_domain->dbx_sub_box_lattice_region.x_low;
  const int lolocaly = p_domain->dbx_sub_box_lattice_region.y_low;
  const int lolocalz = p_domain->dbx_sub_box_lattice_region.z_low;
  const int nlocalx = p_domain->dbx_sub_box_lattice_size[0];
  const int nlocaly = p_domain->dbx_sub_box_lattice_size[1];
  const int nlocalz = p_domain->dbx_sub_box_lattice_size[2];
  const int loghostx = p_domain->dbx_ghost_ext_lattice_region.x_low;
  const int loghosty = p_domain->dbx_ghost_ext_lattice_region.y_low;
  const int loghostz = p_domain->dbx_ghost_ext_lattice_region.z_low;
  const int nghostx = p_domain->dbx_ghost_extended_lattice_size[0];
  const int nghosty = p_domain->dbx_ghost_extended_lattice_size[1];
  const int nghostz = p_domain->dbx_ghost_extended_lattice_size[2];

  h_domain.ghost_size_x = lolocalx - loghostx; // [6]
  h_domain.ghost_size_y = lolocaly - loghosty;
  h_domain.ghost_size_z = lolocalz - loghostz;
  h_domain.box_size_x = nlocalx; // [3]
  h_domain.box_size_y = nlocaly;
  h_domain.box_size_z = nlocalz;
  h_domain.ext_size_x = nghostx; // [0]
  h_domain.ext_size_y = nghosty;
  h_domain.ext_size_z = nghostz;
  h_domain.box_index_start_x = lolocalx;
  h_domain.box_index_start_y = lolocaly;
  h_domain.box_index_start_z = lolocalz;

  // copy data of h_domain to device side.
  // then we can use variable `d_domain` in kernel function
  setDeviceDomain(h_domain);

#ifdef _MD_DEV_MODE
  _hipDeviceDomain sym_domain;
  HIP_CHECK(hipMemcpyFromSymbol(&sym_domain, HIP_SYMBOL(d_domain), sizeof(_hipDeviceDomain)));
  debug_printf("domain on device side: ghost (%d, %d, %d), box (%d, %d, %d), ext (%d, %d, %d), start (%ld, %ld, %ld).\n",
         sym_domain.ghost_size_x, sym_domain.ghost_size_y, sym_domain.ghost_size_z, sym_domain.box_size_x,
         sym_domain.box_size_y, sym_domain.box_size_z, sym_domain.ext_size_x, sym_domain.ext_size_y,
         sym_domain.ext_size_z, sym_domain.box_index_start_x, sym_domain.box_index_start_z,
         sym_domain.box_index_start_z);
#endif
  debug_printf("copy domain metadata done.\n");
  /*
  cout<<nghostx<<endl;//62
  cout<<nghosty<<endl;//31
  cout<<nghostz<<endl;//56
  cout<<nlocalx<<endl;//50
  cout<<nlocaly<<endl;//25
  cout<<nlocalz<<endl;//50
  cout<<lolocalx - loghostx<<endl;//6
  cout<<lolocaly - loghosty<<endl;//3
  cout<<lolocalz - loghostz<<endl;//3
  */
}

void hip_nei_offset_init(const NeighbourIndex<_type_neighbour_index_ele> *nei_offset) {
  // constValue_int[21] = neighbours->nei_half_odd_offsets.size();//114
  // constValue_int[22] = neighbours->nei_half_even_offsets.size();
#ifndef USE_NEWTONS_THIRD_LOW
  size_t nei_odd_size = nei_offset->nei_odd_offsets.size(); // 228 //偏移和原子id是long类型的,不过不影响？
  size_t nei_even_size = nei_offset->nei_even_offsets.size();
  // the data will be used in kernel, need to convert from type NeiOffset to _type_nei_offset_kernel.
  std::vector<_type_nei_offset_kernel> nei_odd(nei_odd_size);
  std::vector<_type_nei_offset_kernel> nei_even(nei_even_size);

  for (size_t i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_odd_offsets[i]; // 一维偏移量索引
  }
  for (size_t i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_even_offsets[i];
  }
#endif

#ifdef USE_NEWTONS_THIRD_LOW
  const size_t nei_odd_size = nei_offset->nei_half_odd_offsets.size();
  const size_t nei_even_size = nei_offset->nei_half_even_offsets.size();

  // the data will be used in kernel, need to convert from type NeiOffset to _type_nei_offset_kernel.
  std::vector<_type_nei_offset_kernel> nei_odd(nei_odd_size);
  std::vector<_type_nei_offset_kernel> nei_even(nei_even_size);

  // sub_box区域内原子的邻居索引（因为x的odd，even，间隙原子等造成的不同，且x,y,z均大于等于0）(各维增量形式->一维增量）
  for (size_t i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_half_odd_offsets[i];
  }
  for (size_t i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_half_even_offsets[i];
  }
#endif

  _type_nei_offset_kernel *d_nei_odd, *d_nei_even;
  HIP_CHECK(hipMalloc((void **)&d_nei_odd, sizeof(_type_nei_offset_kernel) * nei_odd_size));
  HIP_CHECK(hipMalloc((void **)&d_nei_even, sizeof(_type_nei_offset_kernel) * nei_even_size));

  // sort neighbor offset array to reduce branch divergence and better memory coalesced in GPU kernel
  if ((OPT_LEVEL & OPT_SORT_NEIGHBOR) != 0) {
    auto comp_nei = [](_type_nei_offset_kernel a, _type_nei_offset_kernel b) { return (std::abs(a) < std::abs(b)); };
    std::sort(nei_odd.begin(), nei_odd.end(), comp_nei);
    std::sort(nei_even.begin(), nei_even.end(), comp_nei);
  }

  HIP_CHECK(
      hipMemcpy(d_nei_odd, nei_odd.data(), sizeof(_type_nei_offset_kernel) * nei_odd_size, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_nei_even, nei_even.data(), sizeof(_type_nei_offset_kernel) * nei_even_size, hipMemcpyHostToDevice));

  d_nei_offset.nei_odd_size = nei_odd_size;
  d_nei_offset.nei_even_size = nei_even_size;
  d_nei_offset.nei_odd = d_nei_odd;
  d_nei_offset.nei_even = d_nei_even;
  debug_printf("copy neighbor offset done.\n");
}

void hip_pot_init(eam *_pot) {
  auto _pot_types = std::vector<atom_type::_type_atomic_no>{0,1,2};
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);
}

// allocate memory for storage atoms information in device side if d_atoms is nullptr.
void allocDeviceAtomsIfNull() {

  // create double buffers.
  const _type_atom_count atoms_per_layer = h_domain.ext_size_y * h_domain.ext_size_x;
  const _type_atom_count max_block_atom_size = // fixme: buffer size
      ((h_domain.box_size_z - 1) / batches_cli + 1 + 2 * h_domain.ghost_size_z) * atoms_per_layer;
  device_atoms::try_malloc_double_buffers(atoms_per_layer, max_block_atom_size);

}

void hip_eam_rho_calc(eam *pot, _type_atom_list_collection _atoms, double cutoff_radius) {
  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  const db_buffer_data_desc data_desc = db_buffer_data_desc{
      .blocks = batches_cli,
      .data_len = h_domain.box_size_z,
      .eles_per_block_item = h_domain.ext_size_y * h_domain.ext_size_x,
  };
  RhoDoubleBufferImp rhp_double_buffer(stream[0], stream[1], data_desc, device_atoms::fromAtomListColl(_atoms),
                                       device_atoms::fromAtomListColl(_atoms), device_atoms::d_atoms_buffer1,
                                       device_atoms::d_atoms_buffer2, h_domain, d_nei_offset, cutoff_radius);
  rhp_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}

void hip_eam_df_calc(eam *pot, _type_atom_list_collection _atoms, double cutoff_radius) {
#ifndef USE_NEWTONS_THIRD_LOW
  return;
#endif

  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  const db_buffer_data_desc data_desc = db_buffer_data_desc{
      .blocks = batches_cli,
      .data_len = h_domain.box_size_z,
      .eles_per_block_item = h_domain.ext_size_y * h_domain.ext_size_x,
  };
  DfDoubleBufferImp df_double_buffer(stream[0], stream[1], data_desc, device_atoms::fromAtomListColl(_atoms),
                                     device_atoms::fromAtomListColl(_atoms), device_atoms::d_atoms_buffer1,
                                     device_atoms::d_atoms_buffer2, h_domain);
  df_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}

void hip_eam_force_calc(eam *pot, _type_atom_list_collection _atoms, double cutoff_radius) {
  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  const db_buffer_data_desc data_desc = db_buffer_data_desc{
      .blocks = batches_cli,
      .data_len = h_domain.box_size_z,
      .eles_per_block_item = h_domain.ext_size_y * h_domain.ext_size_x,
  };
  ForceDoubleBufferImp force_double_buffer(stream[0], stream[1], data_desc, device_atoms::fromAtomListColl(_atoms),
                                           device_atoms::fromAtomListColl(_atoms), device_atoms::d_atoms_buffer1,
                                           device_atoms::d_atoms_buffer2, h_domain, d_nei_offset, cutoff_radius);
  force_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}
