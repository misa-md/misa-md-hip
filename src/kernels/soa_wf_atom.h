//
// Created by genshen on 2022/6/5.
//

#ifndef MISA_MD_SOA_WF_ATOM_H
#define MISA_MD_SOA_WF_ATOM_H

#include <hip/hip_runtime.h>

#include "../kernel_types.h"
#include "kernels/types/vec3.hpp"

template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename DF,
          typename TARGET>
__global__ void md_nei_itl_wf_atom_soa(const POS_TYPE (*__restrict x)[HIP_DIMENSION], const ATOM_TYPE *types,
                                       DF *__restrict df, TARGET *__restrict target, const INDEX_TYPE atoms_num,
                                       const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                                       const POS_TYPE cutoff_radius);

#endif // MISA_MD_SOA_WF_ATOM_H
