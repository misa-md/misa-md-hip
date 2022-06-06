//
// Created by genshen on 2022/6/3.
//

#ifndef MISA_MD_HIP_HIP_KERNEL_TYPES_H
#define MISA_MD_HIP_HIP_KERNEL_TYPES_H

#include "types/vec3.hpp"

// types definition used in HIP kernels
typedef int _type_atom_index_kernel; // note: on cpu side is long (see type type_atom_index on host side).
typedef int _type_atom_type_kernel;

struct TpModeRho {};
struct TpModeForce {};

constexpr int ModeRho = 0;
constexpr int ModeForce = 2;

#endif // MISA_MD_HIP_HIP_KERNEL_TYPES_H
