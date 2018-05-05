#ifndef TF_SERVER_STORAGE_H_O2RAEQJ1
#define TF_SERVER_STORAGE_H_O2RAEQJ1

#include "util/storage/dense_storage.h"

namespace woops
{
template<typename T>
class TfServerStorage: public DenseStorage<T>
{
public:
    TfServerStorage (size_t size): DenseStorage<T>(size) {}
};
} /* woops */ 


#endif /* end of include guard: TF_SERVER_STORAGE_H_O2RAEQJ1 */
