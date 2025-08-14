#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {  
    
status neg(const tensor_t*, tensor_t*);
status add(const tensor_t*, const tensor_t*, tensor_t*);
status sub(const tensor_t*, const tensor_t*, tensor_t*); 
status mul(const tensor_t*, const tensor_t*, tensor_t*);    
status pow(const tensor_t*, const tensor_t*, tensor_t*);

} // namespace cpu