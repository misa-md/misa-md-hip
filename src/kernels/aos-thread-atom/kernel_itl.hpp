//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_AOS_THREAD_ATOM_KERNEL_ITL_H
#define MISA_MD_AOS_THREAD_ATOM_KERNEL_ITL_H

#include <stdio.h>
#include <stdlib.h>

#include "global_ops.h"
#include "kernels/aos_eam_pair.hpp"
#include "kernels/atom_index.hpp"
#include "kernels/types/hip_kernel_types.h"
#include "md_hip_config.h"

template <int MODE>
__device__ __forceinline__ void nei_interaction(int cur_type, _cuAtomElement &cur_atom, _cuAtomElement &nei_atom,
                                                double _x0, double _y0, double _z0, double &t0, double &t1, double &t2,
                                                double cutoff) {
  const int nei_type = nei_atom.type;
  if (nei_type < 0) {
    return;
  } else {
    const double delx = _x0 - nei_atom.x[0];
    const double dely = _y0 - nei_atom.x[1];
    const double delz = _z0 - nei_atom.x[2];
    const double dist2 = delx * delx + dely * dely + delz * delz;
    if (MODE == ModeRho) {
      NEIGHBOR_PAIR_FUNC(rho)(dist2, cutoff, cur_type, nei_type, cur_atom, nei_atom, t0);
    }
    if (MODE == ModeForce) {
      NEIGHBOR_PAIR_FUNC(force)(dist2, cutoff, delx, dely, delz, cur_type, nei_type, cur_atom, nei_atom, t0, t1, t2);
    }
  }
}

template <typename T, int MODE>
__global__ void itl_atoms_pair(_cuAtomElement *d_atoms, T *_d_result_buf, _hipDeviceNeiOffsets offsets,
                               const _ty_data_block_id start_id, const _ty_data_block_id end_id,
                               const double cutoff_radius) {
  const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // atoms number in this data block.
  const _type_atom_index_kernel atoms_num = (end_id - start_id) * d_domain.box_size_x * d_domain.box_size_y;
  const unsigned int threads_size = hipGridDim_x * hipBlockDim_x;

  // loop all atoms in current data-block
  for (_type_atom_index_kernel atom_id = thread_id; atom_id < atoms_num; atom_id += threads_size) {
    const auto lat =
        AtomIdToLattice<_type_atom_index_kernel, _type_atom_index_kernel, _type_lattice_size>(atom_id, d_domain);
    _cuAtomElement &cur_atom = d_atoms[lat.index]; // load the atom

    const int type0 = cur_atom.type;
    if (type0 < 0) { /* do nothing if it is invalid */
      continue;
    }
    // loop each neighbor atoms, and calculate rho contribution
    const size_t j = (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    double t0 = 0.0, t1 = 0.0, t2 = 0.0; // summation of rho or force
    const double x0 = cur_atom.x[0];
    const double y0 = cur_atom.x[1];
    const double z0 = cur_atom.x[2];
    for (size_t k = 0; k < j; k++) {
      // neighbor can be indexed with odd x or even x
      const _type_nei_offset_kernel offset =
          (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      _cuAtomElement &nei_atom = d_atoms[lat.index + offset]; /* get neighbor atom*/
      nei_interaction<MODE>(type0, cur_atom, nei_atom, x0, y0, z0, t0, t1, t2, cutoff_radius);
    }
    if (MODE == ModeRho) {
#ifdef USE_NEWTONS_THIRD_LOW
      hip_md_interaction_add(&(cur_atom.rho), t0);
#endif
#ifndef USE_NEWTONS_THIRD_LOW
      cur_atom.rho = t0;
#endif
    }
    if (MODE == ModeForce) {
#ifdef USE_NEWTONS_THIRD_LOW
      hip_md_interaction_add(&(cur_atom.f[0]), t0);
      hip_md_interaction_add(&(cur_atom.f[1]), t1);
      hip_md_interaction_add(&(cur_atom.f[2]), t2);
#endif
#ifndef USE_NEWTONS_THIRD_LOW
      cur_atom.f[0] = t0;
      cur_atom.f[1] = t1;
      cur_atom.f[2] = t2;
#endif
    }

#ifndef USE_NEWTONS_THIRD_LOW
    if (MODE == ModeRho) {
      cur_atom.df = hip_pot::hipDEmbedEnergy(type0, cur_atom.rho);
    }
#endif // USE_NEWTONS_THIRD_LOW
  }
}

#endif // MISA_MD_AOS_THREAD_ATOM_KERNEL_ITL_H
