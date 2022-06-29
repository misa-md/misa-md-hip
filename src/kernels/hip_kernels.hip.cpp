#include <algorithm>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_pot_macros.h"

#include "global_ops.h"
#include "kernel_itl.hpp"
#include "md_hip_config.h"

/**
 * @deprecated
 */
__global__ void calc_rho(_cuAtomElement *, double *, _hipDeviceNeiOffsets, int, int, double) { return; }

__global__ void calDf(_cuAtomElement *d_atoms, _ty_data_block_id start_id, _ty_data_block_id end_id) {
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

/**
 * @deprecated
 */
__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {}
