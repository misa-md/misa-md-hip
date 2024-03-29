//
// Created by genshen on 2021/05/13.
//

#ifndef GLOBAL_OPS_H
#define GLOBAL_OPS_H

#include <hip/hip_runtime.h>

#include "kernel_types.h"

// global domain
extern __device__ __constant__ _hipDeviceDomain d_domain;
//__device__ _hipDeviceKernelParm d_kernelParm;

void setDeviceDomain(_hipDeviceDomain h_domain);

#endif // GLOBAL_OPS_H
