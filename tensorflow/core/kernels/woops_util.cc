#include "woops_util.h"

#include <unordered_map>
#include <string>
#include <mutex>

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace woops
{

using GPUDevice = Eigen::GpuDevice;

namespace {
class Context {
public:
    static Context& Get() {
        static Context singleton;
        return singleton;
    }
    
    void RegisterTrainable(const std::string& tablename) {
        int id = table_id_.size();
        table_id_[tablename] = id;
        id_table_[id] = tablename;
    }

    bool CheckTrainable(const std::string& tablename) {
        return table_id_.count(tablename);
    }

    int Tablename2Id(const std::string& tablename) {
        return table_id_.at(tablename);
    }

    std::string& Id2Tablename(int id) {
        return id_table_.at(id);
    }
private:
    std::unordered_map<std::string, int> table_id_;
    std::unordered_map<int, std::string> id_table_;
};
}


template<typename T>
T* CopyFromTensor(tf::OpKernelContext* ctx, const tf::Tensor& tensor) {
    auto flat = tensor.flat<T>();
    auto ret = new T[flat.size()];
    ctx->eigen_device<GPUDevice>().memcpyDeviceToHost(
            ret, flat.data(), sizeof(T) * flat.size());
    return ret;
}
#include <iostream>

void RegisterTrainable(const std::string& tablename) {
    Context::Get().RegisterTrainable(tablename);
}


int Tablename2Id(const std::string& tablename) {
    return Context::Get().Tablename2Id(tablename);
}

std::string& Id2Tablename(int id) {
    return Context::Get().Id2Tablename(id);
}

bool CheckTrainable(const std::string& tablename) {
    return Context::Get().CheckTrainable(tablename);
}


bool CheckWoopsPrefix(const std::string& str) {
    return CheckPrefix("Woops/", str);
}

bool CheckPrefix(const std::string& prefix, const std::string& str) {
    return prefix.size() <= str.size() &&
        std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool CheckEndWith(const std::string& suffix, const std::string& str) {
    return suffix.size() <= str.size() &&
        std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

//bool CheckTrainable(const std::string& name) {
    //return CheckWoopsPrefix(name) &&
        //!CheckEndWith("moving_mean", name) &&
        //!CheckEndWith("moving_variance", name);
//}

void CopySynchronize(tf::OpKernelContext* ctx) {
    ctx->eigen_device<GPUDevice>().synchronize();
}

template float* CopyFromTensor<float>(tf::OpKernelContext* ctx, const tf::Tensor& tensor);
template double* CopyFromTensor<double>(tf::OpKernelContext* ctx, const tf::Tensor& tensor);
    
} /* woops */ 
