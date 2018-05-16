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
            MAYBE_UNUSED const Placement::Partition& partition) override {
        std::lock_guard<std::mutex> lock(this->mu_);
        auto it = bytes.begin();
        while (it != bytes.end()) {
            ParamIndex idx = *reinterpret_cast<const ParamIndex*>(&(*it));
            std::advance(it, sizeof(ParamIndex)/sizeof(Byte));
            this->data_[idx] += *reinterpret_cast<const T*>(&(*it));
            std::advance(it, sizeof(T)/sizeof(Byte));
        }
    }

template<typename U>
friend class TfWorkerStorage;
};
} /* woops */ 

#endif /* end of include guard: TF_APPLY_BUFFER_H_CPUWJKUK */
