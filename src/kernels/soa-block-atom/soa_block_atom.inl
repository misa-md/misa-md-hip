//
// Created by genshen on 2022/7/10.
//

#include "../atom_index.hpp"
#include "md_hip_building_config.h"
#include "../soa_eam_pair.hpp"

template <typename T, typename V, typename ATOM_TYPE, typename INDEX_TYPE> struct NeighborAtomPair {
  INDEX_TYPE nei_index;
  ATOM_TYPE nei_type;
  //  T dist2;
  T delx;
  T dely;
  T delz;
  __device__ __forceinline__ T dist2() const { return delx * delx + dely * dely + delz * delz; }
};

template <typename T, typename V, typename ATOM_TYPE, typename INDEX_TYPE>
inline int soa_block_atom_kernel_shared_size(const _hipDeviceNeiOffsets offsets, const int wf_num) {
  const int nei_size =
      sizeof(NeighborAtomPair<T, V, ATOM_TYPE, INDEX_TYPE>) * std::max(offsets.nei_even_size, offsets.nei_odd_size);
  const int reduce_size = wf_num * sizeof(V);
  return std::max(nei_size, reduce_size);
}

template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename DF,
          typename TARGET, int THREADS_IN_BLOCK>
__global__ void md_nei_itl_block_atom_soa(const POS_TYPE (*__restrict x)[HIP_DIMENSION], const ATOM_TYPE *types,
                                          DF *__restrict df, TARGET *__restrict target, const INDEX_TYPE atoms_num,
                                          const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                                          const POS_TYPE cutoff_radius) {
  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int tid_in_wf = tid % __WAVE_SIZE__;
  const int tid_in_block = tid % THREADS_IN_BLOCK;
  const int wave_id_in_block = tid_in_block / __WAVE_SIZE__;
  constexpr int WAVES_IN_BLOCK = THREADS_IN_BLOCK / __WAVE_SIZE__;

  extern __shared__ int shared_data[];
  NeighborAtomPair<POS_TYPE, V, ATOM_TYPE, INDEX_TYPE> *neighbor_atoms =
      (NeighborAtomPair<POS_TYPE, V, ATOM_TYPE, INDEX_TYPE> *)(shared_data);

  __shared__ int shared_index[1];
  if (tid_in_block == 0) {
    shared_index[0] = 0;
  }
  __syncthreads();

  // load atom.
  const INDEX_TYPE &atom_id = tid / THREADS_IN_BLOCK;
  const auto lat =
      AtomIdToLattice<_type_atom_index_kernel, _type_atom_index_kernel, _type_lattice_size>(atom_id, d_domain);
  // neighbor loop
  const ATOM_TYPE cur_type = types[lat.index];
  const INDEX_TYPE &cur_index = lat.index;
  if (cur_type < 0) {
    return;
  }

  const POS_TYPE x_src[3] = {x[lat.index][0], x[lat.index][1], x[lat.index][2]};
  // loop each neighbor atoms, and calculate rho contribution
  const bool even_offset = (lat.x + d_domain.box_index_start_x) % 2 == 0;
  const size_t nei_len = even_offset ? offsets.nei_even_size : offsets.nei_odd_size;
  const _type_nei_offset_kernel *offset_list = even_offset ? offsets.nei_even : offsets.nei_odd;

  for (size_t k = tid_in_block; k < nei_len; k += THREADS_IN_BLOCK) {
    // neighbor can be indexed with odd x or even x
    const int offset = offset_list[k];
    const INDEX_TYPE nei_index = lat.index + offset;
    const POS_TYPE x_nei[3] = {x[nei_index][0], x[nei_index][1], x[nei_index][2]};
    const ATOM_TYPE nei_type = types[nei_index];
    if (nei_type < 0) {
      continue;
    }
    const POS_TYPE delx = x_src[0] - x_nei[0];
    const POS_TYPE dely = x_src[1] - x_nei[1];
    const POS_TYPE delz = x_src[2] - x_nei[2];
    const POS_TYPE dist2 = delx * delx + dely * dely + delz * delz;
    if (dist2 >= cutoff_radius * cutoff_radius) {
      continue;
    }
    const int i = atomicAdd(shared_index, 1);
    neighbor_atoms[i] = NeighborAtomPair<POS_TYPE, V, ATOM_TYPE, INDEX_TYPE>{nei_index, nei_type, delx, dely, delz};
  }
  __syncthreads();

  const int shared_index_val = shared_index[0];
  V t; // summation of rho or force
  for (int i = tid_in_block; i < shared_index_val; i += THREADS_IN_BLOCK) {
    NeighborAtomPair<POS_TYPE, V, ATOM_TYPE, INDEX_TYPE> nei = neighbor_atoms[i];
    MODE m;
    POT_SUM<MODE, ATOM_TYPE, DF, POS_TYPE, INDEX_TYPE, V>()(m, t, df, cur_index, nei.nei_index, cur_type, nei.nei_type,
                                                            nei.dist2(), nei.delx, nei.dely, nei.delz);
  }
  __syncthreads();

  // reduction to thread 0 in current block.
  V *temp_mem = (V *)(shared_data);
  t.block_wf_reduce<THREADS_IN_BLOCK, WAVES_IN_BLOCK>(tid_in_wf, tid_in_block, wave_id_in_block, temp_mem);
  // store data back.
  if (tid_in_block == 0) {
    target[lat.index] = t;
    if (std::is_same<MODE, TpModeRho>::value) {
#ifndef USE_NEWTONS_THIRD_LOW
      df[lat.index] = hip_pot::hipDEmbedEnergy(cur_type, t.first());
#endif
    }
  }
}
