//
// Created by genshen on 2020/04/27.
//

#ifndef HIP_KERNELS_H
#define HIP_KERNELS_H

#include "hip/hip_runtime.h"
#include "types/pre_define.h"

#define HIP_DIMENSION 3

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

extern __device__ __constant__ _hipDeviceDomain d_domain;

__global__ void calRho(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius);

__global__ void calDf(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets);

__global__ void calForce(_cuAtomElement *d_atoms, _hipDeviceNeiOffsets offsets, double cutoff_radius);

#endif // HIP_KERNELS_H
