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

__global__ void calc_rho(_cuAtomElement *d_atoms, tp_device_rho *_d_rhos, _hipDeviceNeiOffsets offsets,
                         const _type_atom_index_kernel start_id, const _type_atom_index_kernel end_id,
                         double cutoff_radius) {
  const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  // atoms number in this block.
  const _type_atom_index_kernel atoms_num = (end_id - start_id) * d_domain.box_size_x * d_domain.box_size_y;
  const unsigned int threads_size = hipGridDim_x * hipBlockDim_x;

  int type0;
  double x0, y0, z0, delx, dely, delz;
  double dist, rhoTmp;
  int typetemp;               // type of neighbor atoms
  double xtemp, ytemp, ztemp; // position of neighbor atoms
  int offset;

  // loop all atoms in current data-block
  for (_type_atom_index_kernel atom_id = 0; atom_id < atoms_num; atom_id += threads_size) {
    const _type_atom_index_kernel z = atom_id / (d_domain.box_size_x * d_domain.box_size_y);
    const _type_atom_index_kernel y = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) / d_domain.box_size_x;
    const _type_atom_index_kernel x = (atom_id % (d_domain.box_size_x * d_domain.box_size_y)) % d_domain.box_size_x;
    // array index from starting of current data block
    const _type_atom_index_kernel index = _deviceAtom3DIndexToLinear(x, y, z);
    _cuAtomElement &cur_atom = d_atoms[index]; // get the atom

    x0 = cur_atom.x[0];
    y0 = cur_atom.x[1];
    z0 = cur_atom.x[2];
    type0 = cur_atom.type;
    if (type0 < 0) { // do nothing if it is invalid
      continue;
    }

    // loop each neighbor atoms, and calculate rho contribution
    const size_t j = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    for (size_t k = 0; k < j; k++) {
      // neighbor can be index with odd x or even x
      if ((x + d_domain.box_index_start_x) % 2 == 0) {
        offset = offsets.nei_even[k];
      } else {
        offset = offsets.nei_odd[k];
      }
      _cuAtomElement &nei_atom = d_atoms[index + offset]; // get neighbor atom
      xtemp = nei_atom.x[0];
      ytemp = nei_atom.x[1];
      ztemp = nei_atom.x[2];
      typetemp = nei_atom.type;
      if (typetemp < 0) {
        continue;
      } else {
        delx = x0 - xtemp;
        dely = y0 - ytemp;
        delz = z0 - ztemp;
        dist = delx * delx + dely * dely + delz * delz;
        // calculate rho, and update rho of self atom and neighbor atom.
        if (dist < cutoff_radius * cutoff_radius) {
          rhoTmp = hip_pot::hipChargeDensity(typetemp, dist);
          atomicAdd_(&cur_atom.rho, rhoTmp);
          rhoTmp = hip_pot::hipChargeDensity(type0, dist);
          atomicAdd_(&nei_atom.rho, rhoTmp);
        }
      }
    }
  }
}

__global__ void calDf(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets) {
  //三维线程id映射
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (!_deviceIsAtomInBox(x, y, z)) { //判断线程是否越界
    return;
  }

  const _type_atom_index index = _deviceAtom3DIndexToLinear(x, y, z);
  const int type0 = d_atoms[index].type;
  const double rho = d_atoms[index].rho;

  if (type0 < 0) { //间隙原子，什么都不做 todo: put it here?
  } else {
    d_atoms[index].df = hip_pot::hipDEmbedEnergy(type0, rho);
  }
}

__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {
  //__shared__ int num;
  int typetemp;
  double xtemp, ytemp, ztemp; //邻居原子坐标暂存
  double delx, dely, delz;
  size_t offset; //偏移
  double dist, r, p, fpair;
  double recip, phi, phip, psip, z2, z2p;
  double rho_p_from, rho_p_to;
  double df_from, df_to;
  int m;
  int mtemp;
  //三维线程id映射
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  // int i = (z * d_constValue_int[4] + y) * d_constValue_int[3] + x;//一维线程id
  if (!_deviceIsAtomInBox(x, y, z)) { //判断线程是否越界
    return;
  }

  const _type_atom_index index = _deviceAtom3DIndexToLinear(x, y, z);
  double x0 = d_atoms[index].x[0];
  double y0 = d_atoms[index].x[1];
  double z0 = d_atoms[index].x[2];
  int type0 = d_atoms[index].type;

  /*
  if(i==10000){
      debug_printf("f[0]= %f\n",d_atoms[((z+d_constValue_int[8]) * d_constValue_int[1] + y+d_constValue_int[7])
                                  *d_constValue_int[0]+x+d_constValue_int[6]].f[0]);
  }*/
  //线程对应的需要处理的原子（x+ghost_size_x,y+ghost_size_y,z+ghost_size_z)
  if (type0 < 0) { //间隙原子，什么都不做 todo: put it here?
  } else {
    // x分奇偶的邻居索引
    size_t j = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    for (size_t k = 0; k < j; k++) {
      if ((x + d_domain.box_index_start_x) % 2 == 0) { // x分奇偶的邻居索引
        offset = offsets.nei_even[k];
      } else {
        offset = offsets.nei_odd[k];
      }
      const size_t index_nei = _deviceAtom3DIndexToLinear(x, y, z) + offset;
      xtemp = d_atoms[index_nei].x[0]; //在x0的基础上加offset
      ytemp = d_atoms[index_nei].x[1];
      ztemp = d_atoms[index_nei].x[2];
      typetemp = d_atoms[index_nei].type;
      if (typetemp < 0) {
        // todo
      } else { // type和n作为访问插值的偏移索引
        delx = x0 - xtemp;
        dely = y0 - ytemp;
        delz = z0 - ztemp;
        dist = delx * delx + dely * dely + delz * delz;
        if (dist < cutoff_radius * cutoff_radius) {
          //更新该原子受力
          //嵌入能导数
          df_from = d_atoms[index].df;
          df_to = d_atoms[index_nei].df;

          fpair = hip_pot::hipToForce(type0, typetemp, dist, df_from, df_to);
          d_atoms[index].f[0] += delx * fpair;
          d_atoms[index].f[1] += dely * fpair;
          d_atoms[index].f[2] += delz * fpair;
          /*
          d_atoms[((z + d_constValue_int[8]) * d_constValue_int[1] + y + d_constValue_int[7])
                  * d_constValue_int[0] + x + d_constValue_int[6] + offset].f[0] -= delx * fpair;
          d_atoms[((z + d_constValue_int[8]) * d_constValue_int[1] + y + d_constValue_int[7])
                  * d_constValue_int[0] + x + d_constValue_int[6] + offset].f[1] -= dely * fpair;
          d_atoms[((z + d_constValue_int[8]) * d_constValue_int[1] + y + d_constValue_int[7])
                  * d_constValue_int[0] + x + d_constValue_int[6] + offset].f[2] -= delx * fpair;*/
        }
      }
    }
  }
}
