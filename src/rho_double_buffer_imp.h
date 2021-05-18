//
// Created by genshen on 2021/5/18.
//

#ifndef MISA_MD_RHO_DOUBLE_BUFFER_IMP_H
#define MISA_MD_RHO_DOUBLE_BUFFER_IMP_H

#include "double_buffer.h"

/**
 * double buffer implementation for calculating electron density rho.
 */
class RhoDoubleBufferImp : public DoubleBuffer {
  RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len);

  /**
   * implementation of copying data into device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param block_id current data block id.
   */
  void fillBuffer(hipStream_t &stream, const bool left, const int block_id) override;

  /**
   * implementation of fetching data from device buffer
   * @param stream HIP stream to be used for current data block.
   * @param left whether current buffer is left buffer.
   * @param block_id current data block id.
   */
  void fetchBuffer(hipStream_t &stream, const bool left, const int block_id) override;

  /**
   * implementation of performing calculation for the specific data block.
   * @param stream HIP stream to be used for current data block.
   * @param block_id current data block id.
   */
  void calcAsync(hipStream_t &stream, const int block_id) override;
};

#endif // MISA_MD_RHO_DOUBLE_BUFFER_IMP_H
