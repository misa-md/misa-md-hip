//
// Created by genshen on 2022/5/24.
//

#include <hip/hip_runtime.h>

#include "device_atoms.h"
#include "hip_macros.h" // from hip_pot lib

#define HIP_DIMENSION 3

#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS

namespace device_atoms {
  bool db_buf_created = false;
  _type_buffer_desc d_atoms_buffer1 = {.atoms = nullptr};
  _type_buffer_desc d_atoms_buffer2 = {.atoms = nullptr};
} // namespace device_atoms

void device_atoms::try_malloc_double_buffers(const _type_atom_count atoms_per_layer,
                                             const _type_atom_count max_block_atom_size) {
  // allocate 2 buffers
  if (!db_buf_created) {
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1, sizeof(AtomElement) * max_block_atom_size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2, sizeof(AtomElement) * max_block_atom_size));
    db_buf_created = true;
  }
}

#endif // MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
