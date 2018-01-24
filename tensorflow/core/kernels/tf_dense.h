#ifndef TF_DENSE_H_0UZVFQN5
#define TF_DENSE_H_0UZVFQN5

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

#include "util/storage/storage.h"
#include "util/logging.h"

#include <sstream>
#include <iostream>


namespace woops
{
namespace tf = tensorflow;
using namespace perftools::gputools;
template<typename T>
class TfDense: public Storage
{
public:
    TfDense(tf::mutex* mu, tf::Tensor* tensor, Stream* stream);
    void Zerofy() override;
    size_t GetSize() const override;
    const void* Serialize() const override;
    std::map<Hostid, Bytes> Encoding(const Placement::Partitions& partitions) const override;
    void Assign(const void* data, size_t offset = 0, size_t size = -1) override;
    void Update(const void* delta) override;
    std::string ToString() const override;

private:
    tf::mutex* mu_;
    tf::Tensor* tensor_;
    Stream* stream_;
    mutable std::unique_ptr<T[]> cpu_cache_;

    void CopyToCpuCache_() const;
}; 

template<typename T>
TfDense<T>::TfDense(tf::mutex* mu, tf::Tensor* tensor, Stream* stream):
    mu_(mu),
    tensor_(tensor),
    stream_(stream),
    cpu_cache_(new T[tensor_->NumElements()])
{

}

template<typename T>
std::map<Hostid, Bytes> TfDense<T>::Encoding(const Placement::Partitions& partitions) const {
    std::map<Hostid, Bytes> ret;
    CopyToCpuCache_();
    for (auto& server_part: partitions) {
        Hostid server = server_part.first;
        auto& part = server_part.second;
        ret[server] = std::string{(char*)&cpu_cache_[part.begin], (char*)&cpu_cache_[part.end]};
    }
    return ret;
}

template<typename T>
void TfDense<T>::Zerofy() {
}

template<typename T>
void TfDense<T>::Assign(const void* data, size_t offset, size_t size) {
    tf::mutex_lock l(*mu_);
    const auto flat = tensor_->flat<T>();
    if (size == -1) size = flat.size();
    size_t num_bytes = sizeof(T) * size;
    auto device_dst = DeviceMemory<T>::MakeFromByteSize(((T*)flat.data()) + offset, num_bytes);
    stream_->parent()->SynchronousMemcpyH2D(data, num_bytes, &device_dst);
}

template<typename T>
void TfDense<T>::Update(const void* delta) {
}

template<typename T>
const void *TfDense<T>::Serialize() const {
    CopyToCpuCache_();
    return cpu_cache_.get();
}

template<typename T>
size_t TfDense<T>::GetSize() const {
    return 0;
}

template<typename T>
std::string TfDense<T>::ToString() const {
    CopyToCpuCache_();

    std::stringstream ss;
    for(size_t i = 0; i < tensor_->NumElements(); ++i) {
        ss << cpu_cache_[i] << " ";
    }
    
    return ss.str();
}

template<typename T>
void TfDense<T>::CopyToCpuCache_() const {
    tf::mutex_lock l(*mu_);
    const auto flat = tensor_->flat<T>();
    size_t num_bytes = sizeof(T) * flat.size();
    auto device_src = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()), num_bytes);
    stream_->parent()->SynchronousMemcpyD2H(device_src, num_bytes, cpu_cache_.get());
}

} /* woops */ 

#endif /* end of include guard: TF_DENSE_H_0UZVFQN5 */
