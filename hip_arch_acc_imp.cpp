#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>

#include "arch/arch_imp.h"
#include "hip_kernels.h"
#include "hip_macros.h" // from hip_pot lib
#include "hip_pot.h"

#include "global_ops.h"
#include "kernel_wrapper.h"
#include "md_hip_config.h"

//定义线程块各维线程数
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 16

//#define CUDA_ASSERT(x) (assert((x)==hipSuccess))

_cuAtomElement *d_atoms = nullptr; // atoms data on GPU side
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
  size_t nei_odd_size = nei_offset->nei_odd_offsets.size(); // 228 //偏移和原子id是long类型的,不过不影响？
  size_t nei_even_size = nei_offset->nei_even_offsets.size();
  NeiOffset *nei_odd = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_odd_size);
  NeiOffset *nei_even = (NeiOffset *)malloc(sizeof(NeiOffset) * nei_even_size);

  // sub_box区域内原子的邻居索引（因为x的odd，even，间隙原子等造成的不同，且x,y,z均大于等于0）(各维增量形式->一维增量）
  // cout<<nei_odd_size<<"llllllllllllllllllllllllllllllll"<<endl;
  // cout<<neighbours->nei_even_offsets.size()<<"eeeeeeeeeeeeeeeeeeeeeee"<<endl;
  /*
  for (int i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_half_odd_offsets[i];//一维偏移量索引
  }
  for (int i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_half_even_offsets[i];//
  }*/
  for (size_t i = 0; i < nei_odd_size; i++) {
    nei_odd[i] = nei_offset->nei_odd_offsets[i]; //一维偏移量索引
  }
  for (size_t i = 0; i < nei_even_size; i++) {
    nei_even[i] = nei_offset->nei_even_offsets[i];
  }

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
}

void hip_eam_rho_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  debug_printf("calculating rho.\n");
  // CPU端电子云密度等是清零了的
  //邻居晶格点索引处理
  allocDeviceAtomsIfNull();
  //内存拷贝host->device
  const _type_atom_count size = h_domain.ext_size_x * h_domain.ext_size_y * h_domain.ext_size_z;
  HIP_CHECK(hipMemcpy(d_atoms, atoms, sizeof(AtomElement) * size, hipMemcpyHostToDevice));
  //启动核函数计算
  //一个线程块最多1024个线程,16*8*8,如果sub_size_x等于二倍sub_size_y,即为一个正方体，那么线程任务分配大致平均
  dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blockNumber((h_domain.box_size_x + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, // sub_size_x
                   (h_domain.box_size_y + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y,
                   (h_domain.box_size_z + THREADS_PER_BLOCK_Z - 1) / THREADS_PER_BLOCK_Z);
  debug_printf("launching kernel: <<<%d, %d ,%d>>>\n", (h_domain.box_size_x + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X,
         (h_domain.box_size_y + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y,
         (h_domain.box_size_z + THREADS_PER_BLOCK_Z - 1) / THREADS_PER_BLOCK_Z);
  // hipLaunchKernelGGL(calRho, dim3(blockNumber), dim3(threadsPerBlock), 0, 0, d_atoms, d_constValue_int,
  // d_constValue_double,d_nei_odd,d_nei_even);
  __kernel_calRho_wrapper(blockNumber, threadsPerBlock, d_atoms, d_nei_offset, cutoff_radius);
  if (hipSuccess != hipGetLastError()) {
    debug_printf("error\n");
  }
  debug_printf("kernel finished.\n");
  //内存拷贝device->host
  HIP_CHECK(hipMemcpy(atoms, d_atoms, sizeof(AtomElement) * size, hipMemcpyDeviceToHost)); //一个AtomElement104个字节
                                                                                           //输出测试
  // cout<<"helloooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"<<endl;
  //插值后的spline，发现有很多个0,少数10^-6
}

void hip_eam_df_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  debug_printf("calculating df.\n");
  //间隙原子会导致原子信息发生变化，因此需要再次内存拷贝
  allocDeviceAtomsIfNull();
  const _type_atom_count size = h_domain.ext_size_x * h_domain.ext_size_y * h_domain.ext_size_z;
  HIP_CHECK(hipMemcpy(d_atoms, atoms, sizeof(AtomElement) * size, hipMemcpyHostToDevice));
  debug_printf("copy atoms.\n");
  dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blockNumber((h_domain.box_size_x + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, // sub_size_x
                   (h_domain.box_size_y + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y,
                   (h_domain.box_size_z + THREADS_PER_BLOCK_Z - 1) / THREADS_PER_BLOCK_Z);
  assert(d_atoms != nullptr);
  __kernel_calDf_wrapper(blockNumber, threadsPerBlock, d_atoms, d_nei_offset);
  debug_printf("launching kernels.\n");
  if (hipSuccess != hipGetLastError()) {
    debug_printf("launching kernel error.\n");
  }
  HIP_CHECK(hipMemcpy(atoms, d_atoms, sizeof(AtomElement) * size, hipMemcpyDeviceToHost));
  // cout<<sizeof(_cuAtomElement)<<endl<<"hhhhhhhhhhhhhhhhhhhh";
  // cout<<sizeof(AtomElement)<<endl;
  // cout<<"n是iiii"<<constValue_int[12]<<endl;
}

void hip_eam_force_calc(eam *pot, AtomElement *atoms, double cutoff_radius) {
  debug_printf("calculating force.\n");
  allocDeviceAtomsIfNull();
  const _type_atom_count size = h_domain.ext_size_x * h_domain.ext_size_y * h_domain.ext_size_z;
  // HIP_CHECK(hipMemcpy(d_atoms, atoms, sizeof(AtomElement) * size, assert(hipSuccess==hipMemcpyHostToDevice);
  HIP_CHECK(hipMemcpy(d_atoms, atoms, sizeof(AtomElement) * size, hipMemcpyHostToDevice));
  dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
  dim3 blockNumber((h_domain.box_size_x + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X, // sub_size_x
                   (h_domain.box_size_y + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y,
                   (h_domain.box_size_z + THREADS_PER_BLOCK_Z - 1) / THREADS_PER_BLOCK_Z);
  __kernel_calForce_wrapper(blockNumber, threadsPerBlock, d_atoms, d_nei_offset, cutoff_radius);
  if (hipSuccess != hipGetLastError()) {
    debug_printf("launching kernel error.\n");
  }
  // assert(1<2);
  HIP_CHECK(hipMemcpy(atoms, d_atoms, sizeof(AtomElement) * size, hipMemcpyDeviceToHost));
  // hipFree at the end of one step
}
