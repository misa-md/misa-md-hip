//
// Created by genshen on 2021/5/20.
//

#ifndef MISA_MD_HIP_KERNEL_TYPES_H
#define MISA_MD_HIP_KERNEL_TYPES_H

#include <cstddef>
#include "types/pre_define.h"
#include "arch/arch_atom_list_collection.h"

#define HIP_DIMENSION 3

// block id in double buffer
typedef unsigned int _ty_data_block_id;

typedef struct {
  int cal_atoms_num;
  int threads_num;
  int atoms_per_thread;
} _hipDeviceKernelParm;

typedef struct {
  double ele[3];
} Vec3;

typedef double tp_device_rho;
typedef double tp_device_df;
typedef Vec3 tp_device_force;
typedef _type_atom_index _type_atom_index_kernel;

typedef struct {
  long id;                 // atom id.
  int type;                // atom type
  double x[HIP_DIMENSION]; // atom position.
  double v[HIP_DIMENSION]; // atom velocity.
  double f[HIP_DIMENSION]; // atom force.
  double rho;              // electron charge density
  double df;               // embedded energy
} _cuAtomElement;

// size here are all in BCC struct whit x dimension doubled.
typedef struct {
  _type_lattice_size ghost_size_x; // pure ghost size
  _type_lattice_size ghost_size_y;
  _type_lattice_size ghost_size_z;
  _type_lattice_size box_size_x; // simulation box size on this process
  _type_lattice_size box_size_y;
  _type_lattice_size box_size_z;
  _type_lattice_size ext_size_x; // ghost_size + box_size
  _type_lattice_size ext_size_y;
  _type_lattice_size ext_size_z;
  _type_atom_index box_index_start_x; // global start index
  _type_atom_index box_index_start_y;
  _type_atom_index box_index_start_z;
} _hipDeviceDomain;

typedef struct {
  size_t nei_odd_size;
  size_t nei_even_size;
  NeiOffset *nei_odd;
  NeiOffset *nei_even;
} _hipDeviceNeiOffsets;

typedef _type_atom_list_collection _type_dev_atom_list_collection;

#endif // MISA_MD_HIP_KERNEL_TYPES_H
