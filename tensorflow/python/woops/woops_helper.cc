#include"tensorflow/python/woops/woops_helper.h" 

#include <fstream>

#include "tensorflow/core/kernels/woops_util.h"
#include "woops.h"

namespace tensorflow
{
void woops_initialize_from_file(std::string configfile) {
    woops::InitializeFromFile(configfile);
} 

void woops_clock() {
    woops::Clock();
}

void woops_force_sync() {
    woops::ForceSync();
}

void woops_register_trainable(const char* name) {
    woops::RegisterTrainable(name);
}

} /* tensorflow */ 
