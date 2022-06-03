//
// Created by genshen on 2022/5/31.
//

#ifndef MISA_MD_SOA_THREAD_ATOM_H
#define MISA_MD_SOA_THREAD_ATOM_H

#include <hip/hip_runtime.h>

#include "../kernel_types.h"

typedef int tp_domain_size;
typedef struct {
  tp_domain_size box_size_x, box_size_y, box_size_z;
} dev_domain_param;

template <int MODE, typename I, typename T, typename R, typename DF, typename F, typename P>
__global__ void md_nei_itl_soa(const T (*__restrict x)[HIP_DIMENSION], const P *types, R *__restrict rho,
                               DF *__restrict df, F (*__restrict force)[HIP_DIMENSION], const I atoms_num,
                               const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                               const T cutoff_radius);

#endif // MISA_MD_SOA_THREAD_ATOM_H
