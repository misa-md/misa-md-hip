//
// Created by genshen on 2022/6/6.
//

#ifndef MISA_MD_SOA_EAM_PAIR_HPP
#define MISA_MD_SOA_EAM_PAIR_HPP

#include <hip/hip_runtime.h>

#include "common/utils.h"
#include "hip_eam_device.h"
#include "types/hip_kernel_types.h"

template <typename MODE, typename ATOM_TYPE, typename LOAD_TYPE, typename POS_TYPE, typename INDEX_TYPE,
          typename RESULT_TYPE>
struct POT_SUM {
  __device__ __forceinline__ void operator()(const MODE m, RESULT_TYPE &t, const LOAD_TYPE *src_data,
                                             const INDEX_TYPE cur_index, const INDEX_TYPE nei_index,
                                             const ATOM_TYPE cur_type, const ATOM_TYPE nei_type, const POS_TYPE dist2,
                                             const POS_TYPE delta_x, const POS_TYPE delta_y, const POS_TYPE delta_z) {}
};

// class partial specialization.
template <typename ATOM_TYPE, typename LOAD_TYPE, typename POS_TYPE, typename INDEX_TYPE, typename RESULT_TYPE>
struct POT_SUM<TpModeRho, ATOM_TYPE, LOAD_TYPE, POS_TYPE, INDEX_TYPE, RESULT_TYPE> {
  __device__ __forceinline__ void operator()(const TpModeRho m, RESULT_TYPE &t, const LOAD_TYPE *src_data,
                                             const INDEX_TYPE cur_index, const INDEX_TYPE nei_index,
                                             const ATOM_TYPE cur_type, const ATOM_TYPE nei_type, const POS_TYPE dist2,
                                             const POS_TYPE delta_x, const POS_TYPE delta_y, const POS_TYPE delta_z) {
    LOAD_TYPE rhoTmp = hip_pot::hipChargeDensity(nei_type, dist2); // todo LOAD_TYPE is not accuracy.
    t.data += rhoTmp;
#ifdef USE_NEWTONS_THIRD_LOW // todo:
    rhoTmp = hip_pot::hipChargeDensity(cur_type, dist2);
    hip_md_interaction_add(&nei_atom.rho, rhoTmp);
#endif
  }
};

template <typename ATOM_TYPE, typename LOAD_TYPE, typename POS_TYPE, typename INDEX_TYPE, typename RESULT_TYPE>
struct POT_SUM<TpModeForce, ATOM_TYPE, LOAD_TYPE, POS_TYPE, INDEX_TYPE, RESULT_TYPE> {
  __device__ __forceinline__ void operator()(const TpModeForce h, RESULT_TYPE &t, const LOAD_TYPE *src_data,
                                             const INDEX_TYPE cur_index, const INDEX_TYPE nei_index,
                                             const ATOM_TYPE cur_type, const ATOM_TYPE nei_type, const POS_TYPE dist2,
                                             const POS_TYPE delta_x, const POS_TYPE delta_y, const POS_TYPE delta_z) {
    const LOAD_TYPE df_from = src_data[cur_index];
    const LOAD_TYPE df_to = src_data[nei_index];

    const LOAD_TYPE fpair = hip_pot::hipToForce(cur_type, nei_type, dist2, df_from, df_to);
    const LOAD_TYPE fx = delta_x * fpair;
    const LOAD_TYPE fy = delta_y * fpair;
    const LOAD_TYPE fz = delta_z * fpair;
    t.data[0] += fx;
    t.data[1] += fy;
    t.data[2] += fz;
#ifdef USE_NEWTONS_THIRD_LOW // todo:
    hip_md_interaction_add(&(nei_atom.f[0]), -fx);
    hip_md_interaction_add(&(nei_atom.f[1]), -fy);
    hip_md_interaction_add(&(nei_atom.f[2]), -fz);
#endif
  }
};

#endif // MISA_MD_SOA_EAM_PAIR_HPP
