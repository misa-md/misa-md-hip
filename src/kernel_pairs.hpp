//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_KERNEL_PAIRS_H
#define MISA_MD_HIP_KERNEL_PAIRS_H

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_macros.h"
#include "hip_pot_device.h"

#include "kernel_pairs.hpp"
#include "kernel_types.h"
#include "md_hip_config.h"

#define NEIGHBOR_PAIR_FUNC(itl_name) pair##_##itl_name
#define NEIGHBOR_ITL_FUNC(itl_name, ...) calc##_##itl_name

#define NEIGHBOR_PAIR_IMP(itl_name, ...) __device__ __forceinline__ void NEIGHBOR_PAIR_FUNC(itl_name)(__VA_ARGS__)

NEIGHBOR_PAIR_IMP(rho, const double dist2, const double cutoff_radius, const int cur_type, const int nei_type,
                  _cuAtomElement &cur_atom, _cuAtomElement &nei_atom) {
  if (dist2 > cutoff_radius) {
    return;
  }
  double rhoTmp = hip_pot::hipChargeDensity(nei_type, dist2);
  atomicAdd_(&cur_atom.rho, rhoTmp);
  rhoTmp = hip_pot::hipChargeDensity(cur_type, dist2);
  atomicAdd_(&nei_atom.rho, rhoTmp);
}

NEIGHBOR_PAIR_IMP(df, const double dist2, const double cutoff_radius, const int cur_type, const int nei_type,
                  _cuAtomElement &cur_atom, _cuAtomElement &nei_atom) {
  // todo:
}

NEIGHBOR_PAIR_IMP(force, const double dist2, const double cutoff_radius, const int cur_type, const int nei_type,
                  _cuAtomElement &cur_atom, _cuAtomElement &nei_atom) {
  // todo:
}

#endif // MISA_MD_HIP_KERNEL_PAIRS_H
