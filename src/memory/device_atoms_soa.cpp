//
// Created by genshen on 2022/5/27.
//

#include <hip/hip_runtime.h>

#include "device_atoms.h"
#include "hip_pot_macros.h" // from hip_pot lib
#define MAX_GPU_NUM_PER_NODE 4
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
namespace device_atoms {
  bool db_buf_created[MAX_GPU_NUM_PER_NODE] = {false, false, false, false};
  _type_buffer_desc d_atoms_buffer1[MAX_GPU_NUM_PER_NODE];
  _type_buffer_desc d_atoms_buffer2[MAX_GPU_NUM_PER_NODE];
  /*
  _type_buffer_desc d_atoms_buffer1 = {
      .types = nullptr, .x = nullptr, .v = nullptr, .f = nullptr, .rho = nullptr, .df = nullptr};
  _type_buffer_desc d_atoms_buffer2 = {
      .types = nullptr, .x = nullptr, .v = nullptr, .f = nullptr, .rho = nullptr, .df = nullptr};*/
} // namespace device_atoms

void device_atoms::try_malloc_double_buffers(const _type_atom_count atoms_per_layer,
                                             const _type_atom_count max_block_atom_size, int index) {
  // allocate 2 buffers
  if (!db_buf_created[index]) {
    // for buffer1
    const _type_atom_count &size = max_block_atom_size;
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1[index].types, sizeof(_type_atom_type_enum) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1[index].x, sizeof(_type_atom_location[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1[index].f, sizeof(_type_atom_force[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1[index].rho, sizeof(_type_atom_rho) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer1[index].df, sizeof(_type_atom_df) * size));
    // for buffer2
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2[index].types, sizeof(_type_atom_type_enum) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2[index].x, sizeof(_type_atom_location[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2[index].f, sizeof(_type_atom_force[HIP_DIMENSION]) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2[index].rho, sizeof(_type_atom_rho) * size));
    HIP_CHECK(hipMalloc((void **)&d_atoms_buffer2[index].df, sizeof(_type_atom_df) * size));
    db_buf_created[index] = true;
  }
}

#endif // MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
