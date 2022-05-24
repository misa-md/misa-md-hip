//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_KERNEL_ITL_H
#define MISA_MD_HIP_KERNEL_ITL_H

#include <stdio.h>
#include <stdlib.h>

#include "kernel_pairs.hpp"
#include "kernel_types.h"
#include "global_ops.h"
#include "md_hip_config.h"

/**
 * convert a 3d index (x,y,z) to Linear index in atoms array.
 * where x, y, z are the atom coordinate in simulation box (not include ghost region).
 * note: the atom coordinate at x dimension is doubled.
 */
__device__ __forceinline__ _type_atom_index _deviceAtom3DIndexToLinear(const _type_atom_index x,
                                                                       const _type_atom_index y,
                                                                       const _type_atom_index z) {
  return ((z + d_domain.ghost_size_z) * d_domain.ext_size_y + y + d_domain.ghost_size_y) * d_domain.ext_size_x + x +
         d_domain.ghost_size_x;
}

constexpr int ModeRho = 0;
constexpr int ModeForce = 2;

// convert lattice id
#define ATOM_ID_TO_LATTICE(TP, atom_id, d_domain)                                                                      \
  const TP z = (atom_id) / ((d_domain).box_size_x * (d_domain).box_size_y);                                            \
  const TP y = ((atom_id) % ((d_domain).box_size_x * (d_domain).box_size_y)) / (d_domain).box_size_x;                  \
  const TP x = ((atom_id) % ((d_domain).box_size_x * (d_domain).box_size_y)) % (d_domain).box_size_x;                  \
  /* array index from starting of current data block */                                                                \
  const TP index = _deviceAtom3DIndexToLinear(x, y, z);                                                                \
  _cuAtomElement &cur_atom = d_atoms[index]; /* get the atom */                                                        \
                                                                                                                       \
  double x0 = cur_atom.x[0];                                                                                           \
  double y0 = cur_atom.x[1];                                                                                           \
  double z0 = cur_atom.x[2];                                                                                           \
  int type0 = cur_atom.type;                                                                                           \
  if (type0 < 0) { /* do nothing if it is invalid */                                                                   \
    continue;                                                                                                          \
  }

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
    ATOM_ID_TO_LATTICE(_type_atom_index_kernel, atom_id, d_domain);
    // loop each neighbor atoms, and calculate rho contribution
    const size_t j = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    double t0 = 0.0, t1 = 0.0, t2 = 0.0; // summation of rho or force
    for (size_t k = 0; k < j; k++) {
      // neighbor can be index with odd x or even x
      const int offset = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      _cuAtomElement &nei_atom = d_atoms[index + offset]; /* get neighbor atom*/
      nei_interaction<MODE>(type0, cur_atom, nei_atom, x0, y0, z0, t0, t1, t2, cutoff_radius);
    }
    if (MODE == ModeRho) {
      cur_atom.rho = t0;
    }
    if (MODE == ModeForce) {
      cur_atom.f[0] = t0;
      cur_atom.f[1] = t1;
      cur_atom.f[2] = t2;
    }

#ifndef USE_NEWTONS_THIRD_LOW
    if (MODE == ModeRho) {
      cur_atom.df = hip_pot::hipDEmbedEnergy(type0, cur_atom.rho);
    }
#endif // USE_NEWTONS_THIRD_LOW
  }
}

#endif // MISA_MD_HIP_KERNEL_ITL_H
