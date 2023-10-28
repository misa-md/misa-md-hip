//
// Created by lihuizhao on 2023/10/18.
//

#ifndef MISA_MD_BITMAP_SOA_THREAD_ATOM_H
#define MISA_MD_BITMAP_SOA_THREAD_ATOM_H

#include <hip/hip_runtime.h>

#include "kernel_types.h"


template <typename MODE, typename P, typename I, typename T, typename V, typename DF_TYPE, typename OUT_TYPE>
__global__ void md_nei_bitmap_itl_soa(const T (*__restrict x)[HIP_DIMENSION], const P *types, DF_TYPE *__restrict df,
                               OUT_TYPE(*__restrict out), const I atoms_num, const _hipDeviceNeiOffsets offsets,
                               const _hipDeviceDomain domain, const T cutoff_radius,int* bitmap_mem);

#endif // MISA_MD_SOA_THREAD_ATOM_H
