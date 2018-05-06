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
        std::lock_guard<std::mutex> lock(this->mu_);
        std::map<Hostid, Bytes> ret;
        for (auto&& server_part: partitions) {
            Hostid server = server_part.first;
            const Placement::Partition& part = server_part.second;
            ret[server] = Bytes{(Byte*)&this->data_[part.begin], (Byte*)&this->data_[part.end]};
        }
        this->zerofy();
        return ret;
    }
};
} /* woops */ 


#endif /* end of include guard: TF_TRANSMIT_BUFFER_H_YSJOVQJ3 */
