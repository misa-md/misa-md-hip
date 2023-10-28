//
// Created by lihuizhao on 2023/10/17.
//

#include "hip_eam_device.h"

#include "../../global_ops.h"
#include "../atom_index.hpp"
#include "../hip_kernels.h"
#include "../soa_eam_pair.hpp"
#include "kernel_types.h"
#include "kernels/types/hip_kernel_types.h"
#include "kernels/types/vec3.hpp"

/**
 * This function will calculate the interaction of two atoms specified by index @param cur_index and @param nei_index.
 * The result of the interaction will be saved at @param t0.
 * @tparam MODE calculation mode
 * @tparam ATOM_TYPE type of atom type
 * @tparam INDEX_TYPE type of atom index
 * @tparam POS_TYPE type of distance
 * @tparam INP_TYPE type of input data
 * @tparam V type of return value
 */
//FLOPS=60
template <typename MODE, typename ATOM_TYPE, typename INDEX_TYPE, typename POS_TYPE, typename V, typename INP_TYPE,
          typename TARGET>
__device__ __forceinline__ void nei_distance(INP_TYPE *inp, TARGET *target, const INDEX_TYPE cur_index,
                                                    const INDEX_TYPE nei_index, const ATOM_TYPE cur_type,
                                                    const ATOM_TYPE nei_type, const POS_TYPE x0[3],
                                                    const POS_TYPE nei_x[3], V &t0, POS_TYPE cutoff,
                                                    int bitmap_of_single_atom[8],int k) {
  if (nei_type < 0) {
    return;
  }
  const POS_TYPE delx = x0[0] - nei_x[0];
  const POS_TYPE dely = x0[1] - nei_x[1];
  const POS_TYPE delz = x0[2] - nei_x[2];
  const POS_TYPE dist2 = delx * delx + dely * dely + delz * delz;
  if (dist2 >= cutoff * cutoff) {
    return;
  }
  bitmap_of_single_atom[k/32]=bitmap_of_single_atom[k/32]|(1<<(31-k%32));

}

/**
 *
 * @tparam MODE calculation mode: rho/df/force
 * @tparam I type of atom indexing.
 * @tparam T type of position.
 * @tparam P type of types array.
 * @tparam RT result type. It can be vec3 for force calculation, or vec1 type for rho calculation.
 *   Usually, @tparam RT is the same as @tparam OUT_TYPE.
 * @tparam DF_TYPE type of df array for force and rho calculation).
 * @tparam OUT_TYPE type of output array. It can be force array in force calculation, or rho array in rho calculation.
 * @param x atoms position
 * @param types atoms position
 * @param rho rho array.
 * @return
 */
template <typename MODE, typename P, typename I, typename T, typename V, typename DF_TYPE, typename OUT_TYPE>
__global__ void md_nei_bitmap(const T (*__restrict x)[HIP_DIMENSION], const P *types, DF_TYPE *__restrict df,
                               OUT_TYPE(*__restrict out), const I atoms_num, const _hipDeviceNeiOffsets offsets,
                               const _hipDeviceDomain domain, const T cutoff_radius,int* bitmap_mem) {
  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int global_threads = hipGridDim_x * hipBlockDim_x;
  for (I atom_id = tid; atom_id < atoms_num; atom_id += global_threads) {
    // id to xyz index and array index.
    const auto lat =
        AtomIdToLattice<_type_atom_index_kernel, _type_atom_index_kernel, _type_lattice_size>(atom_id, d_domain);

    const P type0 = types[lat.index];
    if (type0 < 0) {
      continue;
    }

    const T x_src[3] = {x[lat.index][0], x[lat.index][1], x[lat.index][2]};
    int bitmap_of_single_atom[8]={0};
    // loop each neighbor atoms, and calculate rho contribution
    //nei_len大概是200
    const size_t nei_len = (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even_size : offsets.nei_odd_size;
    V t0; // summation of rho or force
    for (size_t k = 0; k < nei_len; k++) {//FLOPS=62
      // neighbor can be indexed with odd x or even x
      const _type_nei_offset_kernel offset =
          (lat.x + d_domain.box_index_start_x) % 2 == 0 ? offsets.nei_even[k] : offsets.nei_odd[k];
      const I nei_index = lat.index + offset;
      const T x_nei[3] = {x[nei_index][0], x[nei_index][1], x[nei_index][2]};
      const P nei_type = types[nei_index];
      if (nei_type < 0) {
        continue;
      }
      const T delx = x_src[0] - x_nei[0];
      const T dely = x_src[1] - x_nei[1];
      const T delz = x_src[2] - x_nei[2];
      const T dist2 = delx * delx + dely * dely + delz * delz;
      if (dist2 >= cutoff_radius * cutoff_radius) {
        continue;
      }
      bitmap_of_single_atom[k/32]=bitmap_of_single_atom[k/32]|(1<<(31-k%32));
    }
    for(size_t i=0;i<8;i++){
        bitmap_mem[atom_id*8+i]=bitmap_of_single_atom[i];
    }
    /**
    if(atom_id==0){
      printf("\n赋值后bitmap_mem中的0号原子:");
      for(int i=0;i<8;i++){
        printf("%d ",bitmap_mem[i]);
      }
      printf("\n");
    }

    if(atom_id==0){
      printf("\n0号原子:");
      for(int i=0;i<8;i++){
        printf("%d ",bitmap_of_single_atom[i]);
      }
      printf("\n");
    }
    **/
  }
}

template __global__ void
md_nei_bitmap<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec1, double, _type_d_vec1>(
    const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict df, _type_d_vec1 *__restrict rho,
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const double cutoff_radius,
    int *bitmap_mem);

template __global__ void
md_nei_bitmap<TpModeRho, _type_atom_type_kernel, _type_atom_index_kernel, float, _type_s_vec1, float, _type_s_vec1>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict df, _type_s_vec1 *__restrict rho,
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const float cutoff_radius,
    int *bitmap_mem);

template __global__ void
md_nei_bitmap<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, double, _type_d_vec3, double,
               _type_d_vec3>(const double (*__restrict x)[HIP_DIMENSION], const int *types, double *__restrict df,
                             _type_d_vec3(*__restrict force), const int atoms_num, const _hipDeviceNeiOffsets offsets,
                             const _hipDeviceDomain domain, const double cutoff_radius,
                             int *bitmap_mem);

template __global__ void
md_nei_bitmap<TpModeForce, _type_atom_type_kernel, _type_atom_index_kernel, float, _type_s_vec3, float, _type_s_vec3>(
    const float (*__restrict x)[HIP_DIMENSION], const int *types, float *__restrict df, _type_s_vec3(*__restrict force),
    const int atoms_num, const _hipDeviceNeiOffsets offsets, const _hipDeviceDomain domain, const float cutoff_radius,
    int *bitmap_mem);
