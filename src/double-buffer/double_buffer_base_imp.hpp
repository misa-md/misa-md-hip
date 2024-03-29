//
// Created by genshen on 2021/5/21.
//

#ifndef MISA_MD_HIP_DOUBLE_BUFFER_BASE_IMP_HPP
#define MISA_MD_HIP_DOUBLE_BUFFER_BASE_IMP_HPP

#include <iostream>

#include "double_buffer.h"

typedef struct {
  /**
   * suggested data blocks.
   */
  const DoubleBuffer::tp_data_block_id blocks;
  /**
   * total data length in all data blocks. data length must large or equal then blocks number.
   *
   */
  const _type_lattice_size data_len;
  /**
   * max element number of source data in one block item on host side.
   */
  const _type_lattice_size eles_per_block_item;
} db_buffer_data_desc;

/**
 * double buffer with bass data types support.
 * In the double buffer implementation, logically, in each cycle it will copy data from host to device side,
 * then perform calculate and fetch data from device side to host side.
 *
 * 1. Copy host -> device:
 * In one cycle, it will copy one block of data form host (from source data) to device (device buffer1 or buffer2).
 * In one block, it contains several elements of type @tparam ST.
 * The elements number is calculated by: {block items} * {size_per_block_item} + {ghost_size}.
 * Note, for next block, it will only skip over {block items} * {size_per_block_item} elements.
 * Where, {block items} is current block size, which is also {data_end_index - data_start_index};
 * {size_per_block_item} is the number of data with type @tparam ST in one block item;
 * and {ghost_size} is the size of extended part containing {ghost_size} elements with type @tparam ST.
 *
 * 2. Copy device -> host
 * In one cycle, it will fetch one block of data with type @tparam DT form device to host, and save to destination.
 * In one block, the data elements number of fetching is: {block items} * {size_per_block_item} + {ghost2_size},
 * where {ghost2_size} is the size of extended part containing {ghost2_size} elements with type @tparam DT.
 * fetch-offset is used to skip both device buffer1/buffer2 memory and host destination memory (skip unit: type @tparam
 * DT).
 * The destination address of next block will skip {block items} * {size_per_block_item} elements.
 *
 * @tparam BT type of buffer descriptor
 * @tparam ST type of source data descriptor
 * @tparam DT type of destination data descriptor
 */
template <typename BT, typename ST, typename DT> class DoubleBufferBaseImp : public DoubleBuffer {
public:
  /**
   * @param stream1 stream for buffer 1, which is used for syncing buffer 1.
   * @param stream2 stream for buffer 2, which is used for syncing buffer 2.
   * @param data_desc descriptor of the source data for the whole task to be calculated,
   *        including the blocks number, total block items, data size of each block item.
   * @param copy_ghost_size ghost data (type @tparam ST) size when coping date from host side to device side.
   * @param fetch_ghost_size ghost data (type @tparam DT) size when fetching date from device side to host side.
   * @param fetch_offset offset size applied to both device and host address when fetching.
   * @param h_ptr_src_data the source data pointer on host side.
   * @param h_ptr_des_data the destination data pointer on host side for fetching results on device side.
   * @param d_ptr_device_buf1, d_ptr_device_buf2 two data buffers memory on device side.
   *  In double buffer scheduler, it will copy data from  @param _ptr_src_data to one of these buffers.
   */
  DoubleBufferBaseImp(hipStream_t &stream1, hipStream_t &stream2, const db_buffer_data_desc data_desc,
                      const unsigned int copy_ghost_size, const unsigned int fetch_ghost_size,
                      const unsigned int fetch_offset, ST h_ptr_src_data, DT h_ptr_des_data, BT d_ptr_device_buf1,
                      BT d_ptr_device_buf2)
      : DoubleBuffer(stream1, stream2, data_desc.blocks, data_desc.data_len),
        eles_per_block_item(data_desc.eles_per_block_item), copy_ghost_size(copy_ghost_size),
        fetch_ghost_size(fetch_ghost_size), fetch_offset(fetch_offset), h_ptr_src_data(h_ptr_src_data),
        h_ptr_des_data(h_ptr_des_data), d_ptr_device_buf1(d_ptr_device_buf1), d_ptr_device_buf2(d_ptr_device_buf2){};

  /**
   * implementation of copying data into device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param data_start_index, data_end_index data starting and ending index(not include ending index)
   *   for current data block.
   * @param block_id current data block id.
   */
  void fillBuffer(hipStream_t &stream, const bool left, const tp_block_item_idx data_start_index,
                  const tp_block_item_idx data_end_index, const tp_data_block_id block_id) override {
    const std::size_t size = eles_per_block_item * (data_end_index - data_start_index) + copy_ghost_size;
    // ST *h_p = h_ptr_src_data + eles_per_block_item * data_start_index;
    BT d_p = left ? d_ptr_device_buf1 : d_ptr_device_buf2;
    const std::size_t src_offset = eles_per_block_item * data_start_index;
    copyFromHostToDeviceBuf(stream, d_p, h_ptr_src_data, src_offset, size);
  }

  /**
   * implementation of fetching data from device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param data_start_index, data_end_index data starting and ending index(not include ending index)
   *   for current data block.
   * @param block_id current data block id.
   */
  void fetchBuffer(hipStream_t &stream, const bool left, const tp_block_item_idx data_start_index,
                   const tp_block_item_idx data_end_index, const tp_data_block_id block_id) override {
    const std::size_t size = eles_per_block_item * (data_end_index - data_start_index) + fetch_ghost_size;

    BT dev_p = (left ? d_ptr_device_buf1 : d_ptr_device_buf2);
    const std::size_t src_offset = fetch_offset;
    const std::size_t des_offset = eles_per_block_item * data_start_index + fetch_offset;
    copyFromDeviceBufToHost(stream, h_ptr_des_data, dev_p, src_offset, des_offset, size);
  }

protected:
  // number of elements in one block item, it keeps the same as struct db_buffer_data_desc.eles_per_block_item.
  const unsigned int eles_per_block_item;
  // number of additional element to be copies from host to device side.
  const unsigned int copy_ghost_size;
  // number of additional element to be fetched from device to host side.
  const unsigned int fetch_ghost_size;
  // offset size applied to both device and host address when fetching.
  const unsigned int fetch_offset;
  /**
   * source data array.
   * In MD, it can be lattice atoms in current MPI process (including ghost regions).
   **/
  ST h_ptr_src_data; // host address for coping results from.
  BT d_ptr_device_buf1, d_ptr_device_buf2;
  DT h_ptr_des_data; // host address for fetching results to.

protected:
  /**
   * copy data from source (on host side) to device buffer.
   * @param stream hip stream.
   * @param dest_ptr destination buffer on device side.
   * @param src_ptr source data on host side.
   * @param size the size to be copied.
   */
  virtual void copyFromHostToDeviceBuf(hipStream_t &stream, BT dest_ptr, ST src_ptr, const std::size_t src_offset,
                                       std::size_t size) = 0;

  /**
   * Copy data from device side to host side.
   * @param stream hip stream refrence.
   * @param dest_ptr
   * @param src_ptr
   * @param size
   */
  virtual void copyFromDeviceBufToHost(hipStream_t &stream, DT dest_ptr, BT src_ptr, const std::size_t src_offset,
                                       const std::size_t des_offset, std::size_t size) = 0;
};

#endif // MISA_MD_HIP_DOUBLE_BUFFER_BASE_IMP_HPP
