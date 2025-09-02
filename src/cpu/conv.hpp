#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {  

status conv1d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, const size_t pad, const size_t stride);
status conv2d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, const size_t pad[2], const size_t stride[2]) ;
status conv2d(const tensor_t* signal, const tensor_t* kernel, const tensor_t* bias, tensor_t* dst, const size_t pad[2], const size_t stride[2]);

} // namespace cpu