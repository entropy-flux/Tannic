#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu { 
    
status gemm(const tensor_t*, const tensor_t*, tensor_t*, double alpha = 0.0);  

} // namespace cpu