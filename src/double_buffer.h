//
// Created by genshen on 2021/5/16.
//

#ifndef HIP_DOUBLE_BUFFER_H
#define HIP_DOUBLE_BUFFER_H

#include <hip/hip_runtime.h>

class DoubleBuffer {
public:
  DoubleBuffer(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks, const unsigned int data_len);

  /**
   * schedule double-buffer algorithm.
   */
  void schedule();

  /**
   * Send data to buffer
   * @param left true for "left buffer or buffer 1 will be used".
   * @param block_id block id. It can be less than 0, which is invalid.
   */
  virtual void fillBuffer(hipStream_t &stream, const bool left, const int block_id) = 0;
  void fillBufferWrapper(const int block_id);

  /**
   * fetch data back from buffer
   * @param left true for "left buffer or buffer 1 will be used".
   */
  virtual void fetchBuffer(hipStream_t &stream, const bool left, const int block_id) = 0;
  void fetchBufferWrapper(const int block_id);

  /**
   * wait calculation for the specific block to be finished.
   * @param block_id block id.
   */
  virtual void waitCalc(const int block_id);

  /**
   * wait communication (data copy from host to device or device to host)
   * for the specific block to be finished.
   * Thee communication may include "copy back" of last block (id: block_id - 2)
   * and "copy to" of current block (id: block_id).
   * @param block_id
   */
  virtual void waitComm(const int block_id);

  /**
   * perform calculation asynchronously for a specified block.
   * @param block_id block id.
   */
  virtual void calcAsync(hipStream_t &stream, const int block_id) = 0;

protected:
  // hip stream for the two buffers, which is used for sync.
  hipStream_t &stream1, &stream2;
  const unsigned int blocks;   // number of data block
  const unsigned int data_len; // total data length

  /**
   * get data index range for current data block specified by @param block_i
   * @param block_i block id.
   * @param index_start returned value of start index for current block
   * @param index_end returned value of end index for current block (not include @param index_env).
   */
  void getCurrentDataRange(const unsigned int block_i, unsigned int &index_start, unsigned int &index_end);
};

#endif // HIP_DOUBLE_BUFFER_H
