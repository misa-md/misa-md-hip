//
// Created by genshen on 2022/5/24.
//

#ifndef MISA_MD_HIP_DEVICE_ATOMS_H
#define MISA_MD_HIP_DEVICE_ATOMS_H

#include "arch/arch_atom_list_collection.h"
#include "kernel_types.h"

#define HIP_DIMENSION 3

// atoms data stored on device side.
namespace device_atoms {
  // double buffer for different memory layout.
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  typedef struct {
    _cuAtomElement *atoms;
  } _type_dev_buffer;
#endif
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  typedef struct {
    _type_atom_type_enum *types;
    _type_atom_location (*x)[HIP_DIMENSION];
    _type_atom_velocity (*v)[HIP_DIMENSION];
    _type_atom_force (*f)[HIP_DIMENSION];
    _type_atom_rho *rho;
    _type_atom_df *df;
  } _type_dev_buffer;
#endif

  /**
   * For AoS memory layout, the kernel will write result to its input (input as output).
   */
  typedef _type_dev_buffer _type_buffer_desc; // buffer descriptor with data pointer in it.

  /**
   * Buffers for performing double buffer calculation.
   * note: d_atoms_buffer1 is stored on host side, but d_atoms_buffer1.atoms is on device side (the same to
   * d_atoms_buffer2).
   */
  extern _type_buffer_desc d_atoms_buffer1; // = {.atoms = nullptr};
  extern _type_buffer_desc d_atoms_buffer2; // = {.atoms = nullptr};

  /**
   * If buffer array @var d_atoms_buffer1 and @var d_atoms_buffer2 is not allocated.
   * This function will allocate device memory for the two buffers.
   * @param atoms_per_layer the atom number in each x-y plane layer, including ghost atoms.
   * @param max_block_atom_size the max atom number among all blocks.
   */
  void try_malloc_double_buffers(const _type_atom_count atoms_per_layer, const _type_atom_count max_block_atom_size);
}; // namespace device_atoms

#endif // MISA_MD_HIP_DEVICE_ATOMS_H
