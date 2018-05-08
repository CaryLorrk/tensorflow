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
    TfServerStorage (size_t size): DenseStorage<T>(size) {}
    Bytes Encode() override {
        constexpr int COMPRESSION_RATIO = 100;
        std::lock_guard<std::mutex> lock(this->mu_);
        std::vector<ParamIndex> index(this->data_.size());
        std::iota(index.begin(), index.end(), 0);
        auto&& middle = std::next(index.begin(),
                (this->data_.size() + COMPRESSION_RATIO - 1) / COMPRESSION_RATIO);
        std::partial_sort(index.begin(), middle, index.end(), [this](const T& lhs, const T& rhs) {
            auto&& data = this->data_;
            return std::abs(data[lhs]) > std::abs(data[rhs]);
        });
        Bytes ret;
        for (auto it = index.begin(); it != middle; ++it) {
            const ParamIndex& idx = *it;
            T& val = this->data_[idx];
            ret.append((Byte*)&(idx), (Byte*)(&(idx) + 1));
            ret.append((Byte*)&(val), (Byte*)(&(val) + 1));
            val = 0;
        }
        return ret;
    }
    void Decode(Hostid host, const Bytes& bytes) override {
        if (host == Lib::ThisHost()) return;
        std::lock_guard<std::mutex> lock(this->mu_);
        auto it = bytes.begin();
        while (it != bytes.end()) {
            ParamIndex idx = *reinterpret_cast<const ParamIndex*>(&(*it));
            std::advance(it, sizeof(ParamIndex)/sizeof(Byte));
            this->data_[idx] += *reinterpret_cast<const T*>(&(*it));
            std::advance(it, sizeof(T)/sizeof(Byte));
        }
    }
};
} /* woops */ 


#endif /* end of include guard: TF_SERVER_STORAGE_H_O2RAEQJ1 */
