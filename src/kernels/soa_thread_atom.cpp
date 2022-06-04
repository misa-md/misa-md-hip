//
// Created by genshen on 2022/5/28.
//

#include "hip_eam_device.h"
#include "hip_pot_device.h"

#include "../global_ops.h"
#include "../kernel_types.h"
#include "atom_index.hpp"
#include "hip_kernel_types.h"
#include "hip_kernels.h"
#include "soa_thread_atom.h"

/**
 *
 * @tparam MODE calculation mode
 * @tparam T type of distance
 * @tparam FT type of force
 * @tparam P type of return value
 * @tparam ATOM_TYPE type of atom type
 */
template <int MODE, typename I, typename T, typename V, typename R, typename DF, typename F, typename ATOM_TYPE>
__device__ __forceinline__ void nei_interaction_soa(DF *df, const I cur_index, const I nei_index,
                                                    const ATOM_TYPE cur_type, const ATOM_TYPE nei_type, const T x0[3],
                                                    const T nei_x[3], V &t0, V &t1, V &t2, T cutoff) {
  if (nei_type < 0) {
    return;
  }
  const T delx = x0[0] - nei_x[0];
  const T dely = x0[1] - nei_x[1];
  const T delz = x0[2] - nei_x[2];
  const T dist2 = delx * delx + dely * dely + delz * delz;
  if (dist2 >= cutoff * cutoff) {
    return;
  }
  if (MODE == ModeRho) {
    // NEIGHBOR_PAIR_FUNC(rho)(dist2, cutoff, cur_type, nei_type, cur_atom, nei_atom, t0);
    T rhoTmp = hip_pot::hipChargeDensity(nei_type, dist2);
    atomicAdd_(&t0, rhoTmp);
#ifdef USE_NEWTONS_THIRD_LOW // todo:
    rhoTmp = hip_pot::hipChargeDensity(cur_type, dist2);
    atomicAdd_(&nei_atom.rho, rhoTmp);
#endif
  }
  if (MODE == ModeForce) {
    const DF df_from = df[cur_index];
    const DF df_to = df[nei_index];

    const F fpair = hip_pot::hipToForce(cur_type, nei_type, dist2, df_from, df_to);
    const F fx = delx * fpair;
    const F fy = dely * fpair;
    const F fz = delz * fpair;
    atomicAdd_(&t0, fx);
    atomicAdd_(&t1, fy);
    atomicAdd_(&t2, fz);
#ifdef USE_NEWTONS_THIRD_LOW // todo:
    atomicAdd_(&(nei_atom.f[0]), -fx);
    atomicAdd_(&(nei_atom.f[1]), -fy);
    atomicAdd_(&(nei_atom.f[2]), -fz);
#endif
  }
}

/**
 *
 * @tparam MODE calculation mode: rho/df/force
 * @tparam I type of atom indexing.
 * @tparam T type of position.
 * @tparam R type of rho array.
 * @tparam DF type of df array.
 * @tparam F type of force array.
 * @tparam P type of types array.
 * @tparam MODE calculation mode
 * @param x atoms position
 * @param types atoms position
 * @param rho rho array.
 * @return
 */
template <int MODE, typename I, typename T, typename R, typename DF, typename F, typename P>
__global__ void md_nei_itl_soa(const T (*__restrict x)[HIP_DIMENSION], const P *types, R *__restrict rho,
                               DF *__restrict df, F (*__restrict force)[HIP_DIMENSION], const I atoms_num,
                               const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain,
                               const T cutoff_radius) {
  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int global_threads = hipGridDim_x * hipBlockDim_x;
  for (I atom_id = tid; atom_id < atoms_num; atom_id += global_threads) {
    // id to xyz index and array index.
    const auto lat =
        AtomIdToLattice<_type_atom_index_kernel, _type_atom_index_kernel, _type_lattice_size>(atom_id, d_domain);

    const P type0 = types[lat.index];
    if (type0 < 0) {
      continue;
    }

    const T x_src[3] = {x[lat.index][0], x[lat.index][1], x[lat.index][2]};
    // loop each neighbor atoms, and calculate rho contribution
    const size_t nei_len = (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    T t0 = 0.0, t1 = 0.0, t2 = 0.0; // summation of rho or force
    for (size_t k = 0; k < nei_len; k++) {
      // neighbor can be indexed with odd x or even x
      const int offset = (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      const I nei_index = lat.index + offset;
      const T x_nei[3] = {x[nei_index][0], x[nei_index][1], x[nei_index][2]};
      const P nei_type = types[nei_index];
      nei_interaction_soa<MODE, I, T, T, R, DF, F, P>(df, lat.index, nei_index, type0, nei_type, x_src, x_nei, t0, t1,
                                                      t2, cutoff_radius);
    }
    if (MODE == ModeRho) {
      rho[lat.index] = t0;
#ifndef USE_NEWTONS_THIRD_LOW
      df[lat.index] = hip_pot::hipDEmbedEnergy(type0, t0);
#endif
    }

    if (MODE == ModeForce) {
      force[lat.index][0] = t0;
      force[lat.index][1] = t1;
      force[lat.index][2] = t2;
    }
  }
}

template __global__ void
md_nei_itl_soa<ModeRho, _type_atom_index_kernel, double, double, double, double, _type_atom_type_kernel>(
    const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict rho, double *__restrict df,
    double (*__restrict force)[HIP_DIMENSION], const int atoms_num, const _hipDeviceNeiOffsets offsets,
    const _hipDeviceDomain domain, const double cutoff_radius);

template __global__ void
md_nei_itl_soa<ModeRho, _type_atom_index_kernel, float, float, float, float, _type_atom_type_kernel>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict rho, float *__restrict df,
    float (*__restrict force)[HIP_DIMENSION], const int atoms_num, const _hipDeviceNeiOffsets offsets,
    const _hipDeviceDomain domain, const float cutoff_radius);

template __global__ void
md_nei_itl_soa<ModeForce, _type_atom_index_kernel, double, double, double, double, _type_atom_type_kernel>(
    const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict rho, double *__restrict df,
    double (*__restrict force)[HIP_DIMENSION], const int atoms_num, const _hipDeviceNeiOffsets offsets,
    const _hipDeviceDomain domain, const double cutoff_radius);

template __global__ void
md_nei_itl_soa<ModeForce, _type_atom_index_kernel, float, float, float, float, _type_atom_type_kernel>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict rho, float *__restrict df,
    float (*__restrict force)[HIP_DIMENSION], const int atoms_num, const _hipDeviceNeiOffsets offsets,
    const _hipDeviceDomain domain, const float cutoff_radius);
