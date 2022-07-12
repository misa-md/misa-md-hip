//
// Created by genshen on 2022/05/21.
//

#ifndef OPTIMIZATION_LEVEL_H
#define OPTIMIZATION_LEVEL_H

#include "md_hip_building_config.h"

constexpr int OPT_NONE = 0;
constexpr int OPT_DB_BUF = 1 << 0;          // use double buffer (controlled by cli argument --batchs)
constexpr int OPT_PIN_MEM = 1 << 1;         // use pinned memory
constexpr int OPT_SORT_NEIGHBOR = 1 << 2;   // sort neighbor list, which is usually for wf-atom kernel.

// kernel strategy: must be specified by one of following.
constexpr int KERNEL_STRATEGY_THREAD_ATOM = 1 << 0; // one thread for one atom
constexpr int KERNEL_STRATEGY_WF_ATOM = 1 << 1;     // one wavefront for one atom
constexpr int KERNEL_STRATEGY_BLOCK_ATOM = 1 << 2;  // one block for an atom

constexpr int DEFAULT_OPT_LEVEL = OPT_DB_BUF | OPT_PIN_MEM | OPT_SORT_NEIGHBOR;

// set kernel strategy
#if defined(MD_KERNEL_STRATEGY_WF_ATOM) || defined(MD_KERNEL_STRATEGY_DEFAULT)
constexpr int KERNEL_STRATEGY = KERNEL_STRATEGY_WF_ATOM;
#elif defined(MD_KERNEL_STRATEGY_THREAD_ATOM)
constexpr int KERNEL_STRATEGY = KERNEL_STRATEGY_THREAD_ATOM;
#elif defined(MD_KERNEL_STRATEGY_BLOCK_ATOM)
constexpr int KERNEL_STRATEGY = KERNEL_STRATEGY_BLOCK_ATOM;
#else
#error "no kernel strategy defined"
#endif

// set optimization level
#if defined(MD_OPTIMIZE_OPTION_DEFAULT)
constexpr int OPT_LEVEL = DEFAULT_OPT_LEVEL;
#else
constexpr int OPT_LEVEL = MD_OPTIMIZE_OPTION;
#endif

#endif // OPTIMIZATION_LEVEL_H
