//
// Created by genshen on 2020/04/27.
//

#ifndef HIP_CONFIG_H
#define HIP_CONFIG_H

#include <stdio.h>
#include <stdarg.h>

#include "md_hip_building_config.h"
#include "types/pre_define.h" // MD_DEV_MODE macro this header

//#define USE_NEWTONS_THIRD_LOW
namespace global_config {
  inline bool use_newtons_third_law() {
#ifdef USE_NEWTONS_THIRD_LAW
    return true;
#endif
#ifndef USE_NEWTONS_THIRD_LAW
    return false;
#endif
  }
}


#ifdef __cplusplus
extern "C" {
#endif

#ifdef MD_DEV_MODE
inline int debug_printf(const char *format, ...) {
    int done;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    return done;
}
#endif

#ifndef MD_DEV_MODE
inline int debug_printf(const char *cmd, ...) {
    // leave it empty
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif


#endif // HIP_CONFIG_H
