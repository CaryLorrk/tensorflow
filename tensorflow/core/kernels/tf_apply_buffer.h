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
    void Decode(
            const Bytes& bytes,
            const Placement::Partition& partition) override {
        const T* data = reinterpret_cast<const T*>(bytes.data());
        size_t size = bytes.size() / sizeof(T);
        std::lock_guard<std::mutex> lock(this->mu_);
        this->update(data, size, partition.begin);
    }

template<typename U>
friend class TfClientStorage;
};
} /* woops */ 

#endif /* end of include guard: TF_APPLY_BUFFER_H_CPUWJKUK */
