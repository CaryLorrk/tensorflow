#ifndef WOOPS_HELPER_H_LTDQDNPH
#define WOOPS_HELPER_H_LTDQDNPH

#include <string>

namespace tensorflow
{
void woops_initialize_from_file(std::string configfile);
void woops_clock();
void woops_start();
void woops_register_trainable(const char* name);
} /* tensorflow */ 

#endif /* end of include guard: WOOPS_HELPER_H_LTDQDNPH */
