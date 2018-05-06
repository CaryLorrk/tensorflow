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

#include "woops_tf_compress_apply_buffer.h"

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

    void Zerofy() override;
    Bytes Serialize() const override;
    void Deserialize(const Bytes& bytes) override;
    Bytes Encode() override;
    std::map<Hostid, Bytes> Encode(const Placement::Partitions& partitions) override;
    void Decode(Hostid host, const Bytes& bytes) override;
    void Decode(const Bytes& bytes, const Placement::Partition& partition) override;
    void Assign(const Storage& data) override;
    void Update(const Storage& delta) override;
    std::string ToString() const override;

private:
    tf::mutex* mu_;
    tf::Tensor* tensor_;
    Stream* stream_;
    mutable std::vector<T> cpu_cache_;
    mutable std::mutex cpu_cache_mu_; // lock before mu_

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
void TfClientStorage<T>::Zerofy() {
    LOG(FATAL) << "Unimplemented";
}

template<typename T>
Bytes TfClientStorage<T>::Serialize() const {
    std::lock_guard<std::mutex> cpu_lock{ cpu_cache_mu_ };
    {
        std::lock_guard<tf::mutex> lock(*mu_);
        copy_to_cpu_memory(cpu_cache_.data());
    }
    return Bytes{(Byte*)&cpu_cache_[0], cpu_cache_.size() * sizeof(T)};
}

template<typename T>
void TfClientStorage<T>::Deserialize(const Bytes& bytes) {
    size_t size = bytes.size() / sizeof(T);
    std::lock_guard<tf::mutex> lock(*mu_);
    copy_to_gpu_memory(bytes.data(), size);
}

template<typename T>
Bytes TfClientStorage<T>::Encode() {
    Bytes ret;
    LOG(FATAL) << "Unimplemented";
    return ret;
}

template<typename T>
std::map<Hostid, Bytes> TfClientStorage<T>::Encode(
        MAYBE_UNUSED const Placement::Partitions& partitions) {
    std::map<Hostid, Bytes> ret;
    LOG(FATAL) << "Unimplemented";
    return ret;
}

template<typename T>
void TfClientStorage<T>::Decode(
        MAYBE_UNUSED Hostid host,
        MAYBE_UNUSED const Bytes& bytes) {
    LOG(FATAL) << "Unimplemented";
}

template<typename T>
void TfClientStorage<T>::Decode(
        MAYBE_UNUSED const Bytes& bytes,
        MAYBE_UNUSED const Placement::Partition& partition) {
    LOG(FATAL) << "Unimplemented";
}


template<typename T>
void TfClientStorage<T>::Assign(MAYBE_UNUSED const Storage& data) {
    LOG(FATAL) << "Unimplemented";
}

template<typename T>
void TfClientStorage<T>::Update(MAYBE_UNUSED const Storage& delta) {
    auto&& t_delta = reinterpret_cast<const TfApplyBuffer<T>&>(delta);
    std::lock_guard<std::mutex> delta_lock(t_delta.mu_);
    std::lock_guard<std::mutex> cpu_lock(cpu_cache_mu_);
    std::lock_guard<tf::mutex> lock(*mu_);
    update(t_delta.data_.data(), t_delta.data_.size());
}

template<typename T>
std::string TfClientStorage<T>::ToString() const {
    std::lock_guard<std::mutex> lock(cpu_cache_mu_);
    {
        std::lock_guard<tf::mutex> lock(*mu_);
        copy_to_cpu_memory(cpu_cache_.data());
    }
    std::stringstream ss;
    for(size_t i = 0; i < tensor_->NumElements(); ++i) {
        ss << cpu_cache_[i] << " ";
    }
    
    return ss.str();
}

// caller need to hold a lock of mu_
template<typename T>
void TfClientStorage<T>::copy_to_cpu_memory(void* dst, size_t size, size_t offset) const {
    auto&& flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_src = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyD2H(device_src, bytesize, dst);
}

// caller need to hold a lock of mu_
template<typename T>
void TfClientStorage<T>::copy_to_gpu_memory(const void* src, size_t size, size_t offset) {
    auto&& flat = tensor_->flat<T>();
    if (size == 0) size = flat.size();
    size_t bytesize = size * sizeof(T);
    auto device_dst = DeviceMemory<T>::MakeFromByteSize(static_cast<T*>(flat.data()) + offset, bytesize);
    stream_->parent()->SynchronousMemcpyH2D(src, bytesize, &device_dst);
}

// caller need to hold locks of mu_ and cpu_cache_mu_
template<typename T>
void TfClientStorage<T>::update(const T* data, size_t size, size_t offset) {
    copy_to_cpu_memory(&cpu_cache_[offset], size, offset);
    auto&& first = std::next(cpu_cache_.begin(), offset);
    std::transform(data, data + size, first, first, std::plus<T>());
    copy_to_gpu_memory(&cpu_cache_[offset], size, offset);
}

} /* woops */ 

#endif /* end of include guard: TF_DENSE_H_0UZVFQN5 */
