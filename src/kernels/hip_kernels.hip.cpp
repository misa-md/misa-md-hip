#include <algorithm>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_pot_macros.h"

#include "aos-thread-atom/kernel_itl.hpp"
#include "global_ops.h"
#include "md_hip_config.h"

/**
 * @deprecated
 */
__global__ void calc_rho(_cuAtomElement *, double *, _hipDeviceNeiOffsets, int, int, double) { return; }

__global__ void cal_df_aos(_cuAtomElement *d_atoms, _ty_data_block_id start_id, _ty_data_block_id end_id) {
  const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // atoms number in this data block.
  const _type_atom_index_kernel atoms_num = (end_id - start_id) * d_domain.box_size_x * d_domain.box_size_y;
  const unsigned int threads_size = hipGridDim_x * hipBlockDim_x;

  for (int atom_id = thread_id; atom_id < atoms_num; atom_id += threads_size) {
    const int z = atom_id / (d_domain.box_size_x * d_domain.box_size_y);
    const int y = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) / d_domain.box_size_x;
    const int x = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) % d_domain.box_size_x;
    const _type_atom_index index = _deviceAtom3DIndexToLinear<_type_atom_index_kernel, _type_lattice_size>(
        x, y, z, d_domain.ghost_size_x, d_domain.ghost_size_y, d_domain.ghost_size_z, d_domain.ext_size_x,
        d_domain.ext_size_y);
    const int type0 = d_atoms[index].type;
    const double rho = d_atoms[index].rho;

    if (type0 < 0) { //间隙原子，什么都不做 todo: put it here?
      continue;
    }
    d_atoms[index].df = hip_pot::hipDEmbedEnergy(type0, rho);
  }
}

template <typename P, typename T, typename I>
__global__ void cal_df_soa(const T *__restrict rho, T *__restrict df, const P *__restrict types, const I atoms_num,
                           const _hipDeviceDomain domain) {
  const unsigned int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int global_threads = hipGridDim_x * hipBlockDim_x;
  for (I atom_id = tid; atom_id < atoms_num; atom_id += global_threads) {
    // id to xyz index and array index.
    const auto lat =
        AtomIdToLattice<_type_atom_index_kernel, _type_atom_index_kernel, _type_lattice_size>(atom_id, domain);

    const P type0 = types[lat.index];
    if (type0 < 0) {
      continue;
    }

    const T atom_rho = rho[lat.index];
    df[lat.index] = hip_pot::hipDEmbedEnergy(type0, atom_rho);
  }
}

/**
 * @deprecated
 */
__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {}

template __global__ void cal_df_soa<_type_atom_type_enum, _type_atom_rho, _type_atom_count>(
    const _type_atom_rho *__restrict rho, _type_atom_rho *__restrict df, const _type_atom_type_enum *__restrict types,
    const _type_atom_count atoms_num, const _hipDeviceDomain domain);
