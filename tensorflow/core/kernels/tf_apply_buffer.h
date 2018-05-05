#ifndef TF_APPLY_BUFFER_H_CPUWJKUK
#define TF_APPLY_BUFFER_H_CPUWJKUK

#include "util/storage/dense_storage.h"

namespace woops
{
template<typename T>
class TfApplyBuffer: public DenseStorage<T>
{
public:
    TfApplyBuffer (size_t size): DenseStorage<T>(size) {}

template<typename U>
friend class TfDense;
};
} /* woops */ 

#endif /* end of include guard: TF_APPLY_BUFFER_H_CPUWJKUK */
