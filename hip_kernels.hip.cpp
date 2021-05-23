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

__global__ void calDf(_cuAtomElement *d_atoms) {
  //三维线程id映射
  /*
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (!(x < d_domain.box_size_x && y < d_domain.box_size_y && z < d_domain.box_size_z)) { //判断线程是否越界
    return;
  }*/
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int atoms_num = d_domain.box_size_x * d_domain.box_size_y * d_domain.box_size_z;
  // int atoms_per_thread = 61;//200*200*200
  int atoms_per_thread = 1;
  int threads_num = (atoms_num + atoms_per_thread - 1) / atoms_per_thread;
  if (thread_id >= threads_num) {
    return;
  }
  int atom_id;
  int x, y, z;
  _type_atom_index index;
  int type0;
  double rho;
  for (int i = 0; i < atoms_per_thread; i++) {
    atom_id = thread_id * atoms_per_thread + i;
    if (atom_id >= atoms_num) {
      return;
    }
    /*
    if (id >= d_domain.box_size_x * d_domain.box_size_y * d_domain.box_size_z) { //判断线程是否越界
      return;
    }*/
    z = atom_id / (d_domain.box_size_x * d_domain.box_size_y);
    y = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) / d_domain.box_size_x;
    x = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) % d_domain.box_size_x;
    index = _deviceAtom3DIndexToLinear(x, y, z);
    type0 = d_atoms[index].type;
    rho = d_atoms[index].rho;

    if (type0 < 0) { //间隙原子，什么都不做 todo: put it here?
    } else {
      d_atoms[index].df = hip_pot::hipDEmbedEnergy(type0, rho);
    }
  }
}

/**
 * @deprecated
 */
__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {

}
