//
// Created by genshen on 2021/05/13.
//

#ifndef GLOBAL_OPS_H
#define GLOBAL_OPS_H

#include "hip_kernels.h"

// global domain
extern __device__ __constant__ _hipDeviceDomain d_domain;
//__device__ _hipDeviceKernelParm d_kernelParm;

typedef double tp_device_rho;

#endif // GLOBAL_OPS_H
