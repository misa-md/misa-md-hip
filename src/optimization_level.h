//
// Created by genshen on 2022/05/21.
//

#ifndef OPTIMIZATION_LEVEL_H
#define OPTIMIZATION_LEVEL_H

constexpr int OPT_NONE = 0;
constexpr int OPT_DB_BUF = 1 << 0;          // use double buffer (controlled by cli argument --batchs)
constexpr int OPT_PIN_MEM = 1 << 1;         // use pinned memory
constexpr int OPT_SORT_NEIGHBOR = 1 << 2;   // sort neighbor list, which is usually for wf-atom kernel.

// kernel strategy: must be specified by one of following.
constexpr int KERNEL_STRATEGY_THREAD_ATOM = 1 << 0; // one thread for one atom
constexpr int KERNEL_STRATEGY_WF_ATOM = 1 << 1;     // one wavefront for one atom

constexpr int OPT_LEVEL = OPT_DB_BUF | OPT_PIN_MEM | OPT_SORT_NEIGHBOR;

constexpr int KERNEL_STRATEGY = KERNEL_STRATEGY_WF_ATOM;

#endif // OPTIMIZATION_LEVEL_H
