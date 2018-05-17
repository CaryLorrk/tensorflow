#ifndef TF_SERVER_STORAGE_H_O2RAEQJ1
#define TF_SERVER_STORAGE_H_O2RAEQJ1

#include "util/storage/dense_storage.h"

#include "lib.h"

namespace woops
{
template<typename T>
class TfServerStorage: public DenseStorage<T>
{
public:
    TfServerStorage (Tableid id) {
        auto&& partition = Lib::Placement().GetPartitions(id).at(Lib::ThisHost());
        this->data_.resize(partition.end - partition.begin);
    }

    Bytes Encode() override {
        std::lock_guard<std::mutex> lock(this->mu_);
        Bytes ret = Bytes{(Byte*)this->data_.data(), this->data_.size() * sizeof(T)};
        this->zerofy();
        return ret;
    }

    void Decode(Hostid from, Hostid to, const Bytes& bytes) override {
        if (from == to) return;
        const T* data = reinterpret_cast<const T*>(bytes.data());
        size_t size = bytes.size() / sizeof(T);
        std::lock_guard<std::mutex> lock(this->mu_);
        this->update(data, size);
    }
};
} /* woops */ 


#endif /* end of include guard: TF_SERVER_STORAGE_H_O2RAEQJ1 */
