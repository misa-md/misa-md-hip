//
// Created by genshen on 2021/5/30.
//

#include <cassert>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip_macros.h>

#include "arch/arch_imp.h"

bool hip_create_atoms_mem(void **atoms, std::size_t data_type_size, _type_atom_count size_x, _type_atom_count size_y,
                          _type_atom_count size_z) {
  HIP_CHECK(hipHostMalloc(atoms, size_x * size_y * size_z * data_type_size))
  assert(atoms != nullptr);
  return true;
}

bool hip_release_atoms_mem(void *atoms) {
  hipHostFree(atoms);
  return true;
}
