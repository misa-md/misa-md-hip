//
// Created by genshen on 2022/6/3.
//

#ifndef MISA_MD_HIP_ATOM_INDEX_HPP
#define MISA_MD_HIP_ATOM_INDEX_HPP

#include <hip/hip_runtime.h>

#include "../kernel_types.h"
#include "types/hip_kernel_types.h"

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

#endif // MISA_MD_HIP_ATOM_INDEX_HPP
