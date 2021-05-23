//
// Created by genshen on 2021/5/23.
//

#ifndef MISA_MD_HIP_KERNEL_ITL_H
#define MISA_MD_HIP_KERNEL_ITL_H

#include "kernel_pairs.hpp"
#include "kernel_types.h"

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

template <int MODE>
__global__ void itl_atoms_pair(_cuAtomElement *d_atoms, tp_device_rho *_d_rhos, _hipDeviceNeiOffsets offsets,
                               const _type_atom_index_kernel start_id, const _type_atom_index_kernel end_id,
                               double cutoff_radius) {
  const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // atoms number in this block.
  const _type_atom_index_kernel atoms_num = (end_id - start_id) * d_domain.box_size_x * d_domain.box_size_y;
  const unsigned int threads_size = hipGridDim_x * hipBlockDim_x;

  // loop all atoms in current data-block
  for (_type_atom_index_kernel atom_id = 0; atom_id < atoms_num; atom_id += threads_size) {
    const _type_atom_index_kernel z = atom_id / (d_domain.box_size_x * d_domain.box_size_y);
    const _type_atom_index_kernel y = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) / d_domain.box_size_x;
    const _type_atom_index_kernel x = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) % d_domain.box_size_x;
    // array index from starting of current data block
    const _type_atom_index_kernel index = _deviceAtom3DIndexToLinear(x, y, z);
    _cuAtomElement &cur_atom = d_atoms[index]; // get the atom

    double x0 = cur_atom.x[0];
    double y0 = cur_atom.x[1];
    double z0 = cur_atom.x[2];
    int type0 = cur_atom.type;
    if (type0 < 0) { // do nothing if it is invalid
      continue;
    }

    // loop each neighbor atoms, and calculate rho contribution
    int offset;
    const size_t j = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    for (size_t k = 0; k < j; k++) {
      // neighbor can be index with odd x or even x
      if ((x + d_domain.box_index_start_x) % 2 == 0) {
        offset = offsets.nei_even[k];
      } else {
        offset = offsets.nei_odd[k];
      }
      _cuAtomElement &nei_atom = d_atoms[index + offset]; // get neighbor atom
      const double xtemp = nei_atom.x[0];
      const double ytemp = nei_atom.x[1];
      const double ztemp = nei_atom.x[2];
      int nei_type = nei_atom.type;
      if (nei_type < 0) {
        continue;
      } else {
        const double delx = x0 - xtemp;
        const double dely = y0 - ytemp;
        const double delz = z0 - ztemp;
        const double dist2 = delx * delx + dely * dely + delz * delz;
        if (MODE == ModeRho) {
          NEIGHBOR_PAIR_FUNC(rho)(dist2, cutoff_radius, type0, nei_type, cur_atom, nei_atom);
        }
        if (MODE == ModeForce) {
          NEIGHBOR_PAIR_FUNC(force)(dist2, cutoff_radius, type0, nei_type, cur_atom, nei_atom);
        }
      }
    }
  }
}

#endif // MISA_MD_HIP_KERNEL_ITL_H
