//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_AOS_EAM_PAIR_H
#define MISA_MD_AOS_EAM_PAIR_H

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_pot_macros.h"

#include "common/utils.h"
#include "kernel_types.h"
#include "md_hip_config.h"

#define NEIGHBOR_PAIR_FUNC(itl_name) pair##_##itl_name
#define NEIGHBOR_ITL_FUNC(itl_name, ...) calc##_##itl_name

#define NEIGHBOR_PAIR_IMP(itl_name, ...) __device__ __forceinline__ void NEIGHBOR_PAIR_FUNC(itl_name)(__VA_ARGS__)

NEIGHBOR_PAIR_IMP(rho, const double dist2, const double cutoff_radius, const int cur_type, const int nei_type,
                  _cuAtomElement &cur_atom, _cuAtomElement &nei_atom, double &t0) {
  if (dist2 >= cutoff_radius * cutoff_radius) {
    return;
  }
  double rhoTmp = hip_pot::hipChargeDensity(nei_type, dist2);
  t0 += rhoTmp;
  if (global_config::use_newtons_third_law()) {
    rhoTmp = hip_pot::hipChargeDensity(cur_type, dist2);
    hip_md_interaction_add(&nei_atom.rho, rhoTmp);
  }
}

NEIGHBOR_PAIR_IMP(df, const double dist2, const double cutoff_radius, const int cur_type, const int nei_type,
                  _cuAtomElement &cur_atom, _cuAtomElement &nei_atom) {
  // todo:
}

NEIGHBOR_PAIR_IMP(force, const double dist2, const double cutoff_radius, const double delx, const double dely,
                  const double delz, const int cur_type, const int nei_type, _cuAtomElement &cur_atom,
                  _cuAtomElement &nei_atom, double &t0, double &t1, double &t2) {
  if (dist2 >= cutoff_radius * cutoff_radius) {
    return;
  }
  const double df_from = cur_atom.df;
  const double df_to = nei_atom.df;

  double fpair = hip_pot::hipToForce(cur_type, nei_type, dist2, df_from, df_to);
  const double fx = delx * fpair;
  const double fy = dely * fpair;
  const double fz = delz * fpair;
  t0 += fx;
  t1 += fy;
  t2 += fz;
  if (global_config::use_newtons_third_law()) {
    hip_md_interaction_add(&(nei_atom.f[0]), -fx);
    hip_md_interaction_add(&(nei_atom.f[1]), -fy);
    hip_md_interaction_add(&(nei_atom.f[2]), -fz);
  }
}

#endif // MISA_MD_AOS_EAM_PAIR_H
