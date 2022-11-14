//
// Created by genshen on 2021/5/30.
//

#include <args.hpp>
#include <iostream>
#include <string>

#include "arch/arch_imp.h"
#include "cli.h"
#include "md_hip_config.h"

unsigned int batches_cli = 1;
unsigned int gpu_num_per_node = 1;

args::ValueFlag<unsigned int> *flag_batches;
args::ValueFlag<unsigned int> *flag_gpus_per_node;

void hip_cli_options(args::ArgumentParser &parser) {
  // todo delete
  flag_batches = new args::ValueFlag<unsigned int>(parser, "batches", "batches", {'b', "Batches number"});
  flag_gpus_per_node = new args::ValueFlag<unsigned int>(
      parser, "gpus_per_node", "specify how many gpus used in one node", {'g', "gpus_per_node"});
}

bool hip_cli_options_parse(args::ArgumentParser &) {
  if (*flag_batches) {
    batches_cli = args::get(*flag_batches);
  }
  if (*flag_gpus_per_node) {
    gpu_num_per_node = args::get(*flag_gpus_per_node);
  }

#ifdef USE_NEWTONS_THIRD_LAW
  constexpr bool use_newtons_third_law = true;
#endif
#ifndef USE_NEWTONS_THIRD_LAW
  constexpr bool use_newtons_third_law = false;
#endif
  if (use_newtons_third_law && batches_cli != 1) {
    std::cerr << "Currently, batch number which is larger than 1 is not supported if newton's third law is enabled."
              << std::endl;
    return false;
  }
  return true;
}
