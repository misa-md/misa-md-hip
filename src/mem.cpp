//
// Created by genshen on 2021/5/30.
//

#include <cassert>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip_macros.h>

#include "arch/arch_imp.h"

AtomElement *hip_create_atoms_mem(_type_atom_count size_x, _type_atom_count size_y, _type_atom_count size_z) {
  AtomElement *atoms;
  HIP_CHECK(hipHostMalloc(&atoms, size_x * size_y * size_z * sizeof(AtomElement)))
  assert(atoms != nullptr);
  return atoms;
}

bool hip_release_atoms_mem(AtomElement *atoms) {
  hipHostFree(atoms);
  return true;
}
