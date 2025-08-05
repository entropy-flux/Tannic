#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/resources.h"
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic;

namespace cuda { 
    
status gemm(const tensor_t*, const tensor_t*, tensor_t*, stream_t);  

} // namespace cpu