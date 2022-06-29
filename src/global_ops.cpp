//
// Created by genshen on 2021/05/13.
//

#include <iostream>
#include "global_ops.h"
#include "hip_pot_macros.h" // from hip_pot lib

__device__ __constant__ _hipDeviceDomain d_domain;

void setDeviceDomain(_hipDeviceDomain h_domain) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_domain), &h_domain, sizeof(_hipDeviceDomain)));
}
