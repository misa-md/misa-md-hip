//
// Created by genshen on 2022/6/5.
//

#include "soa_wf_atom.h"
#include "atom_index.hpp"
#include "global_ops.h"
#include "md_hip_building_config.h"
#include "soa_eam_pair.hpp"

template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename DF,
          typename TARGET>
__global__ void md_nei_itl_wf_atom_soa(const POS_TYPE (*__restrict x)[HIP_DIMENSION], const ATOM_TYPE *types,
                                       DF *__restrict df, TARGET *__restrict target, const INDEX_TYPE atoms_num,
                                       const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                                       const POS_TYPE cutoff_radius) {
  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int wf_id = tid / __WAVE_SIZE__;
  const int tid_in_wf = tid % __WAVE_SIZE__;

  // load atom.
  const INDEX_TYPE &atom_id = wf_id;
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
  V t0; // summation of rho or force
  for (size_t k = tid_in_wf; k < nei_len; k += __WAVE_SIZE__) {
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
    MODE m;
    POT_SUM<MODE, ATOM_TYPE, DF, POS_TYPE, INDEX_TYPE, V>()(m, t0, df, cur_index, nei_index, cur_type, nei_type, dist2,
                                                            delx, dely, delz);
  }

  // reduction to thread 0 in current wavefront.
  t0.wf_reduce();
  // store data back.
  if (tid_in_wf == 0) {
    target[lat.index] = t0;
    if (std::is_same<MODE, TpModeRho>::value) {
#ifndef USE_NEWTONS_THIRD_LOW
      df[lat.index] = hip_pot::hipDEmbedEnergy(cur_type, t0.first());
#endif
    }
  }
}

template __global__ void
md_nei_itl_wf_atom_soa<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec1, double,
                       _type_d_vec1>(const double (*__restrict x)[HIP_DIMENSION], const _type_atom_type_kernel *types,
                                     double *__restrict df, _type_d_vec1 *__restrict target,
                                     const _type_atom_index_kernel atoms_num, const _hipDeviceNeiOffsets offsets,
                                     const _hipDeviceDomain domain, const double cutoff_radius);
template __global__ void
md_nei_itl_wf_atom_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
                       _type_d_vec3>(const double (*__restrict x)[HIP_DIMENSION], const _type_atom_type_kernel *types,
                                     double *__restrict df, _type_d_vec3 *__restrict target,
                                     const _type_atom_index_kernel atoms_num, const _hipDeviceNeiOffsets offsets,
                                     const _hipDeviceDomain domain, const double cutoff_radius);
