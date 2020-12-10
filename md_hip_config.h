//
// Created by genshen on 2020/04/27.
//

#ifndef HIP_CONFIG_H
#define HIP_CONFIG_H

#include <stdio.h>  
#include <stdarg.h>
#include "types/pre_define.h" // MD_DEV_MODE macro this header

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MD_DEV_MODE
int debug_printf(const char *format, ...) {
    int done;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    return done;
}
#else
int debug_printf(const char *cmd, ...) {
    // leave it empty
    return 0;
}
#endif 

#ifdef __cplusplus
}
#endif


#endif HIP_CONFIG_H
