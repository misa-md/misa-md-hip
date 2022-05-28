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
  // double buffer and atom list for AOS(array of struct) memory layout.
  typedef struct {
    _cuAtomElement *atoms;
  } _type_atom_list_aos;

  // double buffer and atom list for SOA(struct of array) memory layout.
  typedef struct {
    _type_atom_type_enum *types;
    _type_atom_location (*x)[HIP_DIMENSION];
    _type_atom_velocity (*v)[HIP_DIMENSION];
    _type_atom_force (*f)[HIP_DIMENSION];
    _type_atom_rho *rho;
    _type_atom_df *df;
  } _type_atom_list_soa;

  typedef _type_atom_list_aos _type_dev_buffer_aos;
  typedef _type_atom_list_soa _type_dev_buffer_soa;

  /**
   * For AoS memory layout, the kernel will write result to its input (input as output).
   */
#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_AOS
  typedef _type_dev_buffer_aos _type_buffer_desc; // buffer descriptor with data pointer in it.
  typedef _type_atom_list_aos _type_atom_list_desc;
  inline _type_atom_list_desc fromAtomListColl(_type_atom_list_collection coll) {
    return _type_atom_list_desc{.atoms = coll.atoms};
  }
#endif

#ifdef MD_ATOM_HASH_ARRAY_MEMORY_LAYOUT_SOA
  typedef _type_dev_buffer_soa _type_buffer_desc;
  typedef _type_atom_list_soa _type_atom_list_desc;
  inline _type_atom_list_desc fromAtomListColl(_type_atom_list_collection coll) {
    return _type_atom_list_desc{
//        .id = coll.atom_ids,
        .types = coll.atom_types,
        .x = coll.atom_x,
        .v = coll.atom_v,
        .f = coll.atom_f,
        .rho = coll.atom_rho,
        .df = coll.atom_df,
    };
  }
#endif

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
