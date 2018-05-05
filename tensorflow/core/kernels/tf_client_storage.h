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

#include "tf_apply_buffer.h"

#include <sstream>
#include <iostream>

namespace woops
{
namespace tf = tensorflow;
using namespace perftools::gputools;
template<typename T>
class TfClientStorage: public Storage
{
public:
    TfClientStorage(tf::mutex* mu, tf::Tensor* tensor, Stream* stream);

    void Sync(const Bytes& bytes) override;
    void Zerofy() override;
    Bytes Encode() const override;
    std::map<Hostid, Bytes> Encode(const Placement::Partitions& partitions) override;
    void Decode(const Bytes& bytes, size_t offset = 0) override;
    void Assign(const Storage& data, size_t offset = 0) override;
    void Update(const Storage& delta, size_t offset = 0) override;
    std::string ToString() const override;

private:
    tf::mutex* mu_;
    tf::Tensor* tensor_;
    Stream* stream_;
    mutable std::vector<T> cpu_cache_;
    mutable std::mutex cpu_cache_mu_;

    void copy_to_gpu_memory(const void* src, size_t size = 0, size_t offset = 0);
    void copy_to_cpu_memory(void* dst, size_t size = 0, size_t offset = 0) const;
    void update(const T* data, size_t size, size_t offset = 0);
}; 

template<typename T>
TfClientStorage<T>::TfClientStorage(tf::mutex* mu, tf::Tensor* tensor, Stream* stream):
    mu_(mu),
    tensor_(tensor),
    stream_(stream),
    cpu_cache_(tensor_->NumElements())
{

}

template<typename T>
void TfClientStorage<T>::Sync(const Bytes& bytes) {
    size_t size = bytes.size() / sizeof(T);
    copy_to_gpu_memory(bytes.data(), size);

}

template<typename T>
Bytes TfClientStorage<T>::Encode() const {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    return Bytes{(char*)&cpu_cache_[0], cpu_cache_.size() * sizeof(T)};
}

template<typename T>
std::map<Hostid, Bytes> TfClientStorage<T>::Encode(const Placement::Partitions& partitions) {
    std::map<Hostid, Bytes> ret;
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    for (auto&& server_part: partitions) {
        Hostid server = server_part.first;
        auto&& part = server_part.second;
        ret[server] = Bytes{(char*)&cpu_cache_[part.begin], (char*)&cpu_cache_[part.end]};
    }
    return ret;
}

template<typename T>
void TfClientStorage<T>::Decode(const Bytes& bytes, size_t offset) {
    size_t size = bytes.size() / sizeof(T);
    const T* data = reinterpret_cast<const T*>(bytes.data());
    update(data, size, offset);
}

template<typename T>
void TfClientStorage<T>::Zerofy() {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    std::fill(cpu_cache_.begin(), cpu_cache_.end(), 0);
    copy_to_gpu_memory(cpu_cache_.data());
}

template<typename T>
void TfClientStorage<T>::Assign(const Storage& data, size_t offset) {
    LOG(FATAL) << "Unimplemented";
}

template<typename T>
void TfClientStorage<T>::Update(const Storage& delta, size_t offset) {
    auto&& t_delta = reinterpret_cast<const TfApplyBuffer<T>&>(delta);
    update(t_delta.data_.data(), t_delta.data_.size(), offset);
}

template<typename T>
std::string TfClientStorage<T>::ToString() const {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(cpu_cache_.data());
    std::stringstream ss;
    for(size_t i = 0; i < tensor_->NumElements(); ++i) {
        ss << cpu_cache_[i] << " ";
    }
    
    return ss.str();
}

template<typename T>
void TfClientStorage<T>::copy_to_cpu_memory(void* dst, size_t size, size_t offset) const {
    tf::mutex_lock l(*mu_);
    auto&& flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_src = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyD2H(device_src, bytesize, dst);

}

// the caller needs to hold a lock of cpu_cache_mu_
template<typename T>
void TfClientStorage<T>::copy_to_gpu_memory(const void* src, size_t size, size_t offset) {
    tf::mutex_lock l(*mu_);
    auto&& flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_dst = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyH2D(src, bytesize, &device_dst);
}

template<typename T>
void TfClientStorage<T>::update(const T* data, size_t size, size_t offset) {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    copy_to_cpu_memory(&cpu_cache_[offset], size, offset);
    auto&& begin_it = std::next(cpu_cache_.begin(), offset);
    std::transform(data, data + size, begin_it, begin_it, std::plus<T>());
    copy_to_gpu_memory(&cpu_cache_[offset], size, offset);
}

} /* woops */ 

#endif /* end of include guard: TF_DENSE_H_0UZVFQN5 */
