#ifndef WOOPS_UTIL_H_QYTNED6S
#define WOOPS_UTIL_H_QYTNED6S



#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace woops
{
namespace tf = tensorflow;
template<typename T>
T* CopyFromTensor(tf::OpKernelContext* ctx, const tf::Tensor& tensor);

bool CheckWoopsPrefix(const std::string& str);
bool CheckPrefix(const std::string& prefix, const std::string& str);
bool CheckEndWith(const std::string& sufix, const std::string& str);
void RegisterTrainable(const std::string& tablename);
int Tablename2Id(const std::string& tablename);
std::string& Id2Tablename(int id);
bool CheckTrainable(const std::string& tablename);
void CopySynchronize(tf::OpKernelContext* ctx);

extern template float* CopyFromTensor<float>(tf::OpKernelContext* ctx, const tf::Tensor& tensor);
extern template double* CopyFromTensor<double>(tf::OpKernelContext* ctx, const tf::Tensor& tensor);

} /* woops */ 



#endif /* end of include guard: WOOPS_UTIL_H_QYTNED6S */
