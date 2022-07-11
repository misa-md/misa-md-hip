//
// Created by genshen on 2022/7/11.
//

#ifndef MISA_MD_SOA_BLOCK_ATOM_HPP
#define MISA_MD_SOA_BLOCK_ATOM_HPP

#include <hip/hip_runtime.h>

template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename DF,
          typename TARGET, int THREADS_IN_BLOCK>
__global__ void md_nei_itl_block_atom_soa(const POS_TYPE (*__restrict x)[HIP_DIMENSION], const ATOM_TYPE *types,
                                          DF *__restrict df, TARGET *__restrict target, const INDEX_TYPE atoms_num,
                                          const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                                          const POS_TYPE cutoff_radius);

template <typename T, typename V, typename ATOM_TYPE, typename INDEX_TYPE>
inline int soa_block_atom_kernel_shared_size(const _hipDeviceNeiOffsets offsets);
#include "soa_block_atom.inl"

#endif // MISA_MD_SOA_BLOCK_ATOM_HPP
