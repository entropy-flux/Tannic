#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cpu { 
    
void gemm(const tensor_t*, const tensor_t*, tensor_t*);  

} // namespace cpu