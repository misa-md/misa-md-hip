//
// Created by genshen on 2022/5/24.
//

#include <hip/hip_runtime.h>

#include "device_atoms.h"
#include "hip_macros.h" // from hip_pot lib

#define HIP_DIMENSION 3

namespace device_atoms {
  bool db_buf_created = false;
  _type_atom_list_collection d_atoms = {.atoms = nullptr}; // deprecated
  _type_buffer_desc d_atoms_buffer1 = {.atoms = nullptr};
  _type_buffer_desc d_atoms_buffer2 = {.atoms = nullptr};
  tp_device_rho *d_rhos = nullptr;     // deprecated
  tp_device_force *d_forces = nullptr; // deprecated
} // namespace device_atoms

void device_atoms::try_malloc_double_buffers(const _type_atom_count atoms_per_layer,
                                             const _type_atom_count max_block_atom_size) {
  // allocate 2 buffers
  if (!db_buf_created) {
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1, sizeof(AtomElement) * max_block_atom_size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2, sizeof(AtomElement) * max_block_atom_size));
#endif // MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
    // for buffer1
    const size _type_atom_count = max_block_atom_size;
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1.type, sizeof(_type_atom_type_enum) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1.x, sizeof(_type_atom_location[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1.f, sizeof(_type_atom_force[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1.rho, sizeof(_type_atom_rho) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1.df, sizeof(_type_atom_df) * size));
    // for buffer2
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2.type, sizeof(_type_atom_type_enum) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2.x, sizeof(_type_atom_location[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2.f, sizeof(_type_atom_force[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2.rho, sizeof(_type_atom_rho) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2.df, sizeof(_type_atom_df) * size));
#endif // MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
    db_buf_created = true;
  }
}
