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
    Bytes Encode() const override;
    std::map<Hostid, Bytes> Encode(const Placement::Partitions& partitions) const override;
    void Decode(const Bytes& bytes, size_t offset = 0, DecodingType decoding_type = DecodingType::UPDATE) override;
    void Assign(const Bytes& bytes, size_t offset = 0) override;
    void Update(const Bytes& bytes, size_t offset = 0) override;
    std::string ToString() const override;

private:
    tf::mutex* mu_;
    tf::Tensor* tensor_;
    Stream* stream_;
    mutable std::vector<T> cpu_cache_;
    mutable std::mutex cpu_cache_mu_;

    void copy_to_gpu_memory(const void* src, size_t size = 0, size_t offset = 0);
    void copy_to_cpu_memory(void* dst, size_t size = 0, size_t offset = 0) const;
}; 

template<typename T>
TfDense<T>::TfDense(tf::mutex* mu, tf::Tensor* tensor, Stream* stream):
    mu_(mu),
    tensor_(tensor),
    stream_(stream),
    cpu_cache_(tensor_->NumElements())
{

}

template<typename T>
Bytes TfDense<T>::Encode() const {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    return Bytes{(char*)&cpu_cache_[0], cpu_cache_.size() * sizeof(T)};
}

template<typename T>
std::map<Hostid, Bytes> TfDense<T>::Encode(const Placement::Partitions& partitions) const {
    std::map<Hostid, Bytes> ret;
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    for (auto& server_part: partitions) {
        Hostid server = server_part.first;
        auto& part = server_part.second;
        ret[server] = Bytes{(char*)&cpu_cache_[part.begin], (char*)&cpu_cache_[part.end]};
    }
    return ret;
}

template<typename T>
void TfDense<T>::Decode(const Bytes& bytes, size_t offset, DecodingType decoding_type) {
    switch (decoding_type) {
    case DecodingType::ASSIGN:
        Assign(bytes, offset);
        break;
    case DecodingType::UPDATE:
        Update(bytes, offset);
        break;
    default:
        LOG(FATAL) << "Unkown decoding type.";
    }
}

template<typename T>
void TfDense<T>::Zerofy() {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    std::fill(cpu_cache_.begin(), cpu_cache_.end(), 0);
    copy_to_gpu_memory(cpu_cache_.data());
}

template<typename T>
void TfDense<T>::Assign(const Bytes& bytes, size_t offset) {
    copy_to_gpu_memory(bytes.data(), bytes.size()/sizeof(T), offset);
}

template<typename T>
void TfDense<T>::Update(const Bytes& bytes, size_t offset) {
    size_t size = bytes.size() / sizeof(T);
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(&cpu_cache_[offset], size, offset);
    const T* data = reinterpret_cast<const T*>(bytes.data());
    for (size_t i = offset; i < size; ++i) {
        cpu_cache_[i] += data[i];
    }
    copy_to_gpu_memory(&cpu_cache_[offset], size, offset);
}

template<typename T>
std::string TfDense<T>::ToString() const {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    std::stringstream ss;
    for(size_t i = 0; i < tensor_->NumElements(); ++i) {
        ss << cpu_cache_[i] << " ";
    }
    
    return ss.str();
}

template<typename T>
void TfDense<T>::copy_to_cpu_memory(void* dst, size_t size, size_t offset) const {
    tf::mutex_lock l(*mu_);
    const auto flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_src = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyD2H(device_src, bytesize, dst);

}

template<typename T>
void TfDense<T>::copy_to_gpu_memory(const void* src, size_t size, size_t offset) {
    tf::mutex_lock l(*mu_);
    const auto flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_dst = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyH2D(src, bytesize, &device_dst);
}

} /* woops */ 

#endif /* end of include guard: TF_DENSE_H_0UZVFQN5 */
