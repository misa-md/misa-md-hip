#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h>

#include "hip_eam_device.h"
#include "hip_kernels.h"
#include "hip_macros.h"
#include "hip_pot_device.h"
#include "md_hip_config.h"


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

//核函数,发现如果利用牛顿第三定律确实会出现写写冲突，结果大幅偏差。
__global__ void calRho(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius) {
  int typetemp;
  double xtemp, ytemp, ztemp; //邻居原子坐标暂存
  double delx, dely, delz;
  int offset; //偏移
  double dist, r, p, rhoTmp;
  //三维线程id映射
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t z = blockIdx.z * blockDim.z + threadIdx.z;
  // todo: can you use newton's third low?
  // 必须要对线程的每一维进行限制而不是乘积
  if (!_deviceIsAtomInBox(x, y, z)) { //判断线程是否越界
    return;
  }

  const _type_atom_index index = _deviceAtom3DIndexToLinear(x, y, z);
  _cuAtomElement &cur_atom = d_atoms[index];
  double x0 = cur_atom.x[0];
  double y0 = cur_atom.x[1];
  double z0 = cur_atom.x[2];
  int type0 = cur_atom.type;
  /*
  int id0= d_atoms[((z+d_constValue_int[8]) * d_constValue_int[1] + y+d_constValue_int[7])
                          *d_constValue_int[0]+x+d_constValue_int[6]].id;
  int i = (z * d_constValue_int[4] + y) * d_constValue_int[3] + x;//一维线程id
  //debug_printf("sss");
  if(i<10&&i==id0-1){
      debug_printf("id0=%d\n",id0);
      debug_printf("pos=%d\n",((z+d_constValue_int[8]) * d_constValue_int[1] + y+d_constValue_int[7])
                        *d_constValue_int[0]+x+d_constValue_int[6]);
  }*/
  //线程对应的需要处理的原子（x+ghost_size_x,y+ghost_size_y,z+ghost_size_z)
  /*
  if(i==10000){
      debug_printf("x0=%f\n",x0);
      debug_printf("y0=%f\n",y0);
      debug_printf("z0=%f\n",z0);
      debug_printf("type0=%d\n",type0);
      debug_printf("rho0= %f\n",d_atoms[((z+d_constValue_int[8]) * d_constValue_int[1] + y+d_constValue_int[7])
                                  *d_constValue_int[0]+x+d_constValue_int[6]].rho);
  }else{
      ;
  }*/
  if (type0 < 0) { //间隙原子，什么都不做 todo: put it here?
  } else {
    // x分奇偶的邻居索引,此处应为全局x
    const size_t j = (x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    for (size_t k = 0; k < j; k++) {
      /*
      if(i==10000&&k==0){//第六个时间步为啥输出了8个0而不是4个
          debug_printf("k=%d\n",k);
      }*/
      if ((x + d_domain.box_index_start_x) % 2 == 0) { // x分奇偶的邻居索引
        offset = offsets.nei_even[k];
      } else {
        offset = offsets.nei_odd[k];
      }
      _cuAtomElement &nei_atom = d_atoms[index + offset];
      xtemp = nei_atom.x[0]; //在x0的基础上加offset
      ytemp = nei_atom.x[1];
      ztemp = nei_atom.x[2];
      typetemp = nei_atom.type;
      if (typetemp < 0) {
      } else { // type和n作为访问插值的偏移索引
        delx = x0 - xtemp;
        dely = y0 - ytemp;
        delz = z0 - ztemp;
        dist = delx * delx + dely * dely + delz * delz;
        if (dist < cutoff_radius * cutoff_radius) {
          rhoTmp = hip_pot::hipChargeDensity(typetemp, dist);
          // debug_printf("Here %f\n", rhoTmp);
          // return;
          cur_atom.rho += rhoTmp;
          /*
          if(i==10000){//一次性不能做太多输出,否则一个都不输出
              //debug_printf("xtemp=%f\n",xtemp);
              //debug_printf("ytemp=%f\n",ytemp);
              //debug_printf("ztemp=%f\n",ztemp);
              //debug_printf("typetemp=%d\n",typetemp);
              //debug_printf("dist=%f\n",dist);
              //debug_printf("r=%f\n",r);
              debug_printf("m=%d\n",m);
              debug_printf("p=%f\n",p);
              //debug_printf("d_spline[mtemp*7]=%f\n",d_spline[mtemp*7]);
              //debug_printf("d_spline[mtemp*7+1]=%f\n",d_spline[mtemp*7+1]);
              //debug_printf("d_spline[mtemp*7+2]=%f\n",d_spline[mtemp*7+2]);
              // debug_printf("p=%f\n",p);
              //debug_printf("原子位置= %d\n",((z+d_constValue_int[8]) * d_constValue_int[1] +
          y+d_constValue_int[7])*d_constValue_int[0]+x+d_constValue_int[6]);
          debug_printf("rhoTmp= %f\n",rhoTmp);
          debug_printf("rho=%f\n",d_atoms[((z+d_constValue_int[8]) * d_constValue_int[1] +
          y+d_constValue_int[7])*d_constValue_int[0]+x+d_constValue_int[6]].rho);
          }*/
          //更新邻居原子电子云密度//会不会产生写冲突呢？
          /*
          r = sqrt(dist);
          //p=r*invDx+1.0;
          p = r * d_constValue_double[1 + type0] + 1.0;
          m = (int) p;
          //m=max(1,min(m,(rho_n-1)));
          m = max(1, min(m, (d_constValue_int[9 + type0] - 1)));
          p -= m;
          p = min(p, 1.0);
          if (type0 == 0) {
              mtemp = m;
          } else if (type0 == 1) {
              mtemp = m + d_constValue_int[9];
          } else {
              mtemp = m + d_constValue_int[9] + d_constValue_int[10];
          }
          rhoTmp = ((d_spline[mtemp*7+3] * p + d_spline[mtemp*7+4]) * p + d_spline[mtemp*7+5]) * p +
                   d_spline[mtemp*7+6];
          d_atoms[((z+d_constValue_int[8]) * d_constValue_int[1] + y+d_constValue_int[7])
                  *d_constValue_int[0]+x+d_constValue_int[6]+offset].rho+=rhoTmp;*/
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
