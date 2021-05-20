//
// Created by genshen on 2021/5/20.
//

#ifndef MISA_MD_HIP_KERNEL_TYPES_H
#define MISA_MD_HIP_KERNEL_TYPES_H

#include "types/pre_define.h"

#define HIP_DIMENSION 3


typedef double tp_device_rho;

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

#endif // MISA_MD_HIP_KERNEL_TYPES_H
