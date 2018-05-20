#ifndef TF_TRANSMIT_BUFFER_H_YSJOVQJ3
#define TF_TRANSMIT_BUFFER_H_YSJOVQJ3

#include "util/storage/dense_storage.h"

namespace woops
{
template<typename T>
class TfTransmitBuffer: public DenseStorage<T>
{
public:
    TfTransmitBuffer (size_t size): DenseStorage<T>(size) {}
    std::map<Hostid, Bytes> Encode(const Placement::Partitions& partitions) override {
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
        
        auto&& kv = partitions.begin();
        std::map<Hostid, Bytes> ret;
        for (auto it = index.begin(); it != middle; ++it) {
            const ParamIndex& idx = *it;
            T& val = this->data_[idx];

            while (idx >= kv->second.end) ++kv;
            Hostid server = kv->first;
            ret[server].append((Byte*)&(idx), (Byte*)(&(idx) + 1));
            ret[server].append((Byte*)&(val), (Byte*)(&(val) + 1));
            val = 0;
        }
        return ret;
    }
};
} /* woops */ 


#endif /* end of include guard: TF_TRANSMIT_BUFFER_H_YSJOVQJ3 */
