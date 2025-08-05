#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {  
    
status neg(tensor_t const*, tensor_t*);
status add(tensor_t const*, tensor_t const*, tensor_t*);
status sub(tensor_t const*, tensor_t const*, tensor_t*); 
status mul(tensor_t const*, tensor_t const*, tensor_t*);    

} // namespace cpu