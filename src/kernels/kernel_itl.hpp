//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_KERNEL_ITL_H
#define MISA_MD_HIP_KERNEL_ITL_H

#include <stdio.h>
#include <stdlib.h>

#include "global_ops.h"
#include "hip_kernel_types.h"
#include "kernel_pairs.hpp"
#include "md_hip_config.h"

/**
 * convert a 3d index (x,y,z) to Linear index in atoms array.
 * where x, y, z are the atom coordinate in simulation box (not include ghost region).
 * note: the atom coordinate at x dimension is doubled.
 */
template <typename ATOM_INDEX_TYPE, typename DOMAIN_SIZE_TYPE>
__device__ __forceinline__ _type_atom_index _deviceAtom3DIndexToLinear(
    const ATOM_INDEX_TYPE x, const ATOM_INDEX_TYPE y, const ATOM_INDEX_TYPE z, const DOMAIN_SIZE_TYPE ghost_size_x,
    const DOMAIN_SIZE_TYPE ghost_size_y, const DOMAIN_SIZE_TYPE ghost_size_z, const DOMAIN_SIZE_TYPE ext_size_x,
    const DOMAIN_SIZE_TYPE ext_size_y) {
  return ((z + ghost_size_z) * ext_size_y + y + ghost_size_y) * ext_size_x + x + ghost_size_x;
}

constexpr int ModeRho = 0;
constexpr int ModeForce = 2;

template <typename AT, typename TP> struct Id2Lat {
  AT x;     // x coordinate.
  TP index; // linear index for load atom.
};

/**
 * calculate the index for loading atom data from device memory by atom id.
 * @tparam AT atom index type.
 * @tparam TP atom/lattice coordinate type.
 * @tparam LT lattice coordinate type. It is usually be the same as @tparam TP.
 */
template <typename AT, typename TP, typename LT>
__device__ __forceinline__ Id2Lat<AT, TP> AtomIdToLattice(AT atom_id, _hipDeviceDomain d_domain) {
  const AT z = (atom_id) / ((d_domain).box_size_x * (d_domain).box_size_y);
  const AT y = ((atom_id) % ((d_domain).box_size_x * (d_domain).box_size_y)) / (d_domain).box_size_x;
  const AT x = ((atom_id) % ((d_domain).box_size_x * (d_domain).box_size_y)) % (d_domain).box_size_x;
  /* array index from starting of current data block */
  const TP index = _deviceAtom3DIndexToLinear<TP, LT>(x, y, z, d_domain.ghost_size_x, d_domain.ghost_size_y,
                                                      d_domain.ghost_size_z, d_domain.ext_size_x, d_domain.ext_size_y);
  Id2Lat<AT, TP> ret = {.x = x, .index = index};
  return ret;
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
      const int offset = (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      _cuAtomElement &nei_atom = d_atoms[lat.index + offset]; /* get neighbor atom*/
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
