//
// Created by genshen on 2022/5/28.
//

#include "hip_eam_device.h"

#include "../global_ops.h"
#include "../kernel_types.h"
#include "atom_index.hpp"
#include "hip_kernel_types.h"
#include "hip_kernels.h"
#include "kernels/types/vec3.hpp"
#include "soa_eam_pair.hpp"
#include "soa_thread_atom.h"

/**
 * This function will calculate the interaction of two atoms specified by index @param cur_index and @param nei_index.
 * The result of the interaction will be saved at @param t0.
 * @tparam MODE calculation mode
 * @tparam ATOM_TYPE type of atom type
 * @tparam INDEX_TYPE type of atom index
 * @tparam POS_TYPE type of distance
 * @tparam INP_TYPE type of input data
 * @tparam V type of return value
 */
template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename INP_TYPE>
__device__ __forceinline__ void
nei_interaction_soa(INP_TYPE *inp, const INDEX_TYPE cur_index, const INDEX_TYPE nei_index, const ATOM_TYPE cur_type,
                    const ATOM_TYPE nei_type, const POS_TYPE x0[3], const POS_TYPE nei_x[3], V &t0, POS_TYPE cutoff) {
  if (nei_type < 0) {
    return;
  }
  const POS_TYPE delx = x0[0] - nei_x[0];
  const POS_TYPE dely = x0[1] - nei_x[1];
  const POS_TYPE delz = x0[2] - nei_x[2];
  const POS_TYPE dist2 = delx * delx + dely * dely + delz * delz;
  if (dist2 >= cutoff * cutoff) {
    return;
  }
  MODE h;
  POT_SUM<MODE, ATOM_TYPE, INP_TYPE, POS_TYPE, INDEX_TYPE, V>()(h, t0, inp, cur_index, nei_index, cur_type, nei_type,
                                                                dist2, delx, dely, delz);
}

/**
 *
 * @tparam MODE calculation mode: rho/df/force
 * @tparam I type of atom indexing.
 * @tparam T type of position.
 * @tparam P type of types array.
 * @tparam RT result type. It can be vec3 for force calculation, or vec1 type for rho calculation.
 *   Usually, @tparam RT is the same as @tparam OUT_TYPE.
 * @tparam DF_TYPE type of df array for force and rho calculation).
 * @tparam OUT_TYPE type of output array. It can be force array in force calculation, or rho array in rho calculation.
 * @param x atoms position
 * @param types atoms position
 * @param rho rho array.
 * @return
 */
template <typename MODE, typename P, typename I, typename T, typename V, typename DF_TYPE, typename OUT_TYPE>
__global__ void md_nei_itl_soa(const T (*__restrict x)[HIP_DIMENSION], const P *types, DF_TYPE *__restrict df,
                               OUT_TYPE(*__restrict out), const I atoms_num, const _hipDeviceNeiOffsets offsets,
                               const _hipDeviceDomain domain, const T cutoff_radius) {
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
    V t0; // summation of rho or force
    for (size_t k = 0; k < nei_len; k++) {
      // neighbor can be indexed with odd x or even x
      const _type_nei_offset_kernel offset =
          (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      const I nei_index = lat.index + offset;
      const T x_nei[3] = {x[nei_index][0], x[nei_index][1], x[nei_index][2]};
      const P nei_type = types[nei_index];
      nei_interaction_soa<MODE, P, I, T, V, DF_TYPE>(df, lat.index, nei_index, type0, nei_type, x_src, x_nei, t0,
                                                     cutoff_radius);
    }
    out[lat.index] = t0; // or: t0.store_to(out, lat.index);

    if (std::is_same<MODE, TpModeRho>::value) {
#ifndef USE_NEWTONS_THIRD_LOW
      df[lat.index] = hip_pot::hipDEmbedEnergy(type0, t0.first());
#endif
    }
    // todo: newton's third law
  }
}

template __global__ void
md_nei_itl_soa<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec1, double, _type_d_vec1>(
    const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict df, _type_d_vec1 *__restrict rho,
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const double cutoff_radius);

template __global__ void
md_nei_itl_soa<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, float, _type_s_vec1, float, _type_s_vec1>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict df, _type_s_vec1 *__restrict rho,
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const float cutoff_radius);

template __global__ void
md_nei_itl_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
               _type_d_vec3>(const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict df,
                             _type_d_vec3(*__restrict force), const int atoms_num, const _hipDeviceNeiOffsets offsets,
                             const _hipDeviceDomain domain, const double cutoff_radius);

template __global__ void
md_nei_itl_soa<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, float, _type_s_vec3, float, _type_s_vec3>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict df, _type_s_vec3(*__restrict force),
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const float cutoff_radius);
