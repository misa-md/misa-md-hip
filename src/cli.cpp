//
// Created by genshen on 2021/5/30.
//

#include <args.hpp>
#include <iostream>
#include <string>

#include "arch/arch_imp.h"
#include "cli.h"

unsigned int batches_cli = 1;

args::ValueFlag<unsigned int> *flag_batches;

void hip_cli_options(args::ArgumentParser &parser) {
  // todo delete
  flag_batches = new args::ValueFlag<unsigned int>(parser, "batches", "batches", {'b', "Batches number"});
}

bool hip_cli_options_parse(args::ArgumentParser &) {
  if (*flag_batches) {
    batches_cli = args::get(*flag_batches);
    return true;
  }
  return true;
}
