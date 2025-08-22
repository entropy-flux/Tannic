#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"
#include "runtime/streams.h" 

using namespace tannic;

namespace cuda {  
      
status conv1d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, stream_t stream, const size_t pad, const size_t stride);
status conv2d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, stream_t stream, const size_t pad[2], const size_t stride[2]);

} // namespace cuda