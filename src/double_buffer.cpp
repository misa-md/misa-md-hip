//
// Created by genshen on 2021/5/16.
//

#include "double_buffer.h"
#include <iostream>

DoubleBuffer::DoubleBuffer(hipStream_t &_stream1, hipStream_t &_stream2, const unsigned int blocks,
                           const unsigned int data_len)
    : stream1(_stream1), stream2(_stream2), blocks(blocks), data_len(data_len) {
  if (blocks <= 0) {
    std::cerr << "invalid block number." << std::endl; // todo: abort program
    return;
  }
  if (data_len < blocks) {
    std::cerr << "invalid block number and data length." << std::endl;
    return;
  }
}

void DoubleBuffer::schedule() {
  // fill data to buffer 1
  fillBufferWrapper(0); // copy with async
  for (int i = 0; i < blocks; i++) {
    waitCalc(i - 1);           // wait calculation finishing of buffer 1 if i is odd, otherwise, buffer 2.
    fetchBufferWrapper(i - 1); // copy data back from buffer 1 if i is odd, otherwise, buffer 2.
    fillBufferWrapper(i + 1);  // fill next block data to buffer 1 if i is odd, otherwise, buffer 2.
    // wait buffer 2 coping finishing (It can also wait another buffer's calculation),
    // if i is odd, otherwise, buffer 1.
    waitComm(i);
    calcAsync(i % 2 == 0 ? stream1 : stream2, i); // calculate buffer 2 if i is odd, otherwise, buffer 1.
  }
  // fetch data from buffer 2 or buffer 1, if necessary
  waitCalc(blocks - 1);
  fetchBufferWrapper(blocks - 1);
}

void DoubleBuffer::getCurrentDataRange(const unsigned int block_i, unsigned int &index_start, unsigned int &index_end) {
  const unsigned int n_per_block = data_len / blocks;
  index_start = n_per_block * block_i;
  index_end = std::min(index_start + n_per_block, data_len);
}

void DoubleBuffer::fillBufferWrapper(const int block_id) {
  const bool is_left_buffer = (block_id % 2 == 0);
  unsigned int data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);
  fillBuffer(is_left_buffer ? stream1 : stream2, is_left_buffer, block_id);
}

void DoubleBuffer::fetchBufferWrapper(const int block_id) {
  const bool is_left_buffer = (block_id % 2 == 0);
  unsigned int data_start_index = 0, data_end_index = 0;
  getCurrentDataRange(block_id, data_start_index, data_end_index);
  fetchBuffer(is_left_buffer ? stream1 : stream2, is_left_buffer, block_id);
}

void DoubleBuffer::waitCalc(const int block_id) {
  if (block_id < 0 || block_id >= blocks) {
    return;
  }
  if (block_id % 2 == 0) {
    hipStreamSynchronize(stream1);
  } else {
    hipStreamSynchronize(stream2);
  }
}

void DoubleBuffer::waitComm(const int block_id) {
  if (block_id < 0 || block_id >= blocks) {
    return;
  }
  if (block_id % 2 == 0) {
    hipStreamSynchronize(stream1);
  } else {
    hipStreamSynchronize(stream2);
  }
}
