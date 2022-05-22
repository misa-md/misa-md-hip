//
// Created by genshen on 2022/05/21.
//

#ifndef OPTIMIZATION_LEVEL_H
#define OPTIMIZATION_LEVEL_H

constexpr int OPT_NONE = 0;
constexpr int OPT_DB_BUF = 1 << 0;          // use double buffer (controlled by cli argument --batchs)
constexpr int OPT_PIN_MEM = 1 << 1;         // use pinned memory
constexpr int OPT_LEVEL = OPT_DB_BUF | OPT_PIN_MEM;

#endif // OPTIMIZATION_LEVEL_H
