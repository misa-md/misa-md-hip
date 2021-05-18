//
// Created by genshen on 2021/5/18.
//

#include "rho_double_buffer_imp.h"

RhoDoubleBufferImp::RhoDoubleBufferImp(hipStream_t &stream1, hipStream_t &stream2, const unsigned int blocks,
                                       const unsigned int data_len)
    : DoubleBuffer(stream1, stream2, blocks, data_len) {}

void RhoDoubleBufferImp::fillBuffer(hipStream_t &stream, const bool left, const int block_id) {}

void RhoDoubleBufferImp::fetchBuffer(hipStream_t &stream, const bool left, const int block_id) {}

void RhoDoubleBufferImp::calcAsync(hipStream_t &stream, const int block_id) {

}
