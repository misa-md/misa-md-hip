#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>

#include "arch/arch_imp.h"
#include "hip_kernels.h"
#include "hip_macros.h" // from hip_pot lib
#include "hip_pot.h"

#include "cli.h"
#include "double-buffer/df_double_buffer_imp.h"
#include "double-buffer/force_double_buffer_imp.h"
#include "double-buffer/rho_double_buffer_imp.h"
#include "global_ops.h"
#include "kernel_wrapper.h"
#include "md_hip_config.h"

//定义线程块各维线程数
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 16

//#define CUDA_ASSERT(x) (assert((x)==hipSuccess))

_cuAtomElement *d_atoms = nullptr; // atoms data on GPU side
_cuAtomElement *d_atoms_buffer1 = nullptr, *d_atoms_buffer2 = nullptr;
tp_device_rho *d_rhos = nullptr;
tp_device_force *d_forces = nullptr;

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
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_domain), &h_domain, sizeof(_hipDeviceDomain)));

#ifdef MD_DEV_MODE
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

void hip_nei_offset_init(const NeighbourIndex<AtomElement> *nei_offset) {
  // constValue_int[21] = neighbours->nei_half_odd_offsets.size();//114
  // constValue_int[22] = neighbours->nei_half_even_offsets.size();
#ifndef USE_NEWTONS_THIRD_LOW
  size_t nei_odd_size = nei_offset->nei_odd_offsets.size(); // 228 //偏移和原子id是long类型的,不过不影响？
  size_t nei_even_size = nei_offset->nei_even_offsets.size();
  NeiOffset *nei_odd = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_odd_size);
  NeiOffset *nei_even = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_even_size);

  for (size_t i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_odd_offsets[i]; //一维偏移量索引
  }
  for (size_t i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_even_offsets[i];
  }
#endif

#ifdef USE_NEWTONS_THIRD_LOW
  const size_t nei_odd_size = nei_offset->nei_half_odd_offsets.size();
  const size_t nei_even_size = nei_offset->nei_half_even_offsets.size();
  NeiOffset *nei_odd = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_odd_size); // todo delete
  NeiOffset *nei_even = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_even_size);

  for (size_t i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_half_odd_offsets[i];
  }
  for (size_t i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_half_even_offsets[i];
  }
#endif

  NeiOffset *d_nei_odd, *d_nei_even;
  HIP_CHECK(hipMalloc((void **)&d_nei_odd, sizeof(NeiOffset) * nei_odd_size));
  HIP_CHECK(hipMalloc((void **)&d_nei_even, sizeof(NeiOffset) * nei_even_size));

  HIP_CHECK(hipMemcpy(d_nei_odd, nei_odd, sizeof(NeiOffset) * nei_odd_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_nei_even, nei_even, sizeof(NeiOffset) * nei_even_size, hipMemcpyHostToDevice));

  d_nei_offset.nei_odd_size = nei_odd_size;
  d_nei_offset.nei_even_size = nei_even_size;
  d_nei_offset.nei_odd = d_nei_odd;
  d_nei_offset.nei_even = d_nei_even;
  debug_printf("copy neighbor offset done.\n");
}

void hip_pot_init(eam *_pot) {
  auto _pot_types = std::vector<atom_type::_type_atomic_no>{26, 29, 28};
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);
}

// allocate memory for storage atoms information in device side if d_atoms is nullptr.
void allocDeviceAtomsIfNull() {
  if (d_atoms == nullptr) {
    const _type_atom_count size = h_domain.ext_size_x * h_domain.ext_size_y * h_domain.ext_size_z;
    HIP_CHECK(hipMalloc((void **)&d_atoms, sizeof(AtomElement) * size)); // fixme: GPU存得下么
  }

  // allocate 2 buffers
  if (d_atoms_buffer1 == nullptr || d_atoms_buffer1 == nullptr) {
    const _type_atom_count atoms_per_layer = h_domain.ext_size_y * h_domain.ext_size_x;
    const _type_atom_count max_block_atom_size =
        ((h_domain.box_size_z - 1) / batches_cli + 1 + 2 * h_domain.ghost_size_z) * atoms_per_layer;
    if (d_atoms_buffer1 == nullptr) {
      HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1, sizeof(AtomElement) * max_block_atom_size * batches_cli))
    }
    if (d_atoms_buffer2 == nullptr) {
      HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2, sizeof(AtomElement) * max_block_atom_size * batches_cli))
    }
  }

  if (d_rhos == nullptr) {
    const _type_atom_count size_ = h_domain.ext_size_z * h_domain.ext_size_y * h_domain.ext_size_x;
    HIP_CHECK(hipMalloc(&d_rhos, size_ * sizeof(tp_device_rho)))
    HIP_CHECK(hipMemset(d_rhos, 0, size_ * sizeof(tp_device_rho)))
    HIP_CHECK(hipMalloc(&d_forces, size_ * sizeof(tp_device_force)))
    HIP_CHECK(hipMemset(d_forces, 0, size_ * sizeof(tp_device_force)))
  }
}

void hip_eam_rho_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  RhoDoubleBufferImp rhp_double_buffer(stream[0], stream[1], batches_cli, h_domain.box_size_z, atoms, d_atoms_buffer1,
                                       d_atoms_buffer2, d_rhos, h_domain, d_nei_offset, cutoff_radius);
  rhp_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}

void hip_eam_df_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  DfDoubleBufferImp df_double_buffer(stream[0], stream[1], batches_cli, h_domain.box_size_z, atoms, d_atoms_buffer1,
                                     d_atoms_buffer2, d_rhos, h_domain);
  df_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}

void hip_eam_force_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  hipStream_t stream[2];
  for (int i = 0; i < 2; i++) {
    hipStreamCreate(&(stream[i]));
  }
  allocDeviceAtomsIfNull();
  ForceDoubleBufferImp force_double_buffer(stream[0], stream[1], batches_cli, h_domain.box_size_z, atoms,
                                           d_atoms_buffer1, d_atoms_buffer2, d_forces, h_domain, d_nei_offset,
                                           cutoff_radius);
  force_double_buffer.schedule();
  for (int i = 0; i < 2; i++) {
    hipStreamDestroy(stream[i]);
  }
}
