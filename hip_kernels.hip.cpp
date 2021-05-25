#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h>

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_macros.h"
#include "hip_pot_device.h"
#include "md_hip_config.h"
#include "src/global_ops.h"


/**convert a 3d index (x,y,z) to Linear index in atoms array.
 *where x, y, z are the atom coordinate in simulation box (not include ghost region).
 * note: the atom coordinate at x dimension is doubled.
 */
inline __device__ _type_atom_index _deviceAtom3DIndexToLinear(const _type_atom_index x, const _type_atom_index y,
                                                              const _type_atom_index z) {
  return ((z + d_domain.ghost_size_z) * d_domain.ext_size_y + y + d_domain.ghost_size_y) * d_domain.ext_size_x + x +
         d_domain.ghost_size_x;
}

/**
 * Check whether the atoms assigned to current thread is in the simulation box.
 * \param x, y, z: the atom coordinate in simulation region of current MPI process.
 * \return true is In, falsr is Out.
 */
inline __device__ bool _deviceIsAtomInBox(_type_atom_index x, _type_atom_index y, _type_atom_index z) {
  return x < d_domain.box_size_x && y < d_domain.box_size_y && z < d_domain.box_size_z;
}

/**
 * @deprecated
 */
__global__ void calc_rho(_cuAtomElement *, double *, _hipDeviceNeiOffsets, long, long, double) { return; }

__global__ void calDf(_cuAtomElement *d_atoms, _ty_data_block_id start_id, _ty_data_block_id end_id) {
  const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // atoms number in this data block.
  const _type_atom_index_kernel atoms_num = (end_id - start_id) * d_domain.box_size_x * d_domain.box_size_y;
  const unsigned int threads_size = hipGridDim_x * hipBlockDim_x;

  for (int atom_id = thread_id; atom_id < atoms_num; atom_id += threads_size) {
    const int z = atom_id / (d_domain.box_size_x * d_domain.box_size_y);
    const int y = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) / d_domain.box_size_x;
    const int x = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) % d_domain.box_size_x;
    const _type_atom_index index = _deviceAtom3DIndexToLinear(x, y, z);
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
__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {

}
