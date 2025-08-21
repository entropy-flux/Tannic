#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/resources.h"
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic;

namespace cuda { 
    
status gemm(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream, double alpha = 0.0);

} // namespace cpu