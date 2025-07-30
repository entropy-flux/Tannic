#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cuda {  
    
void neg(device_t const*, tensor_t const*, tensor_t*);
void add(device_t const*, tensor_t const*, tensor_t const*, tensor_t*);
void sub(device_t const*, tensor_t const*, tensor_t const*, tensor_t*); 
void mul(device_t const*, tensor_t const*, tensor_t const*, tensor_t*);    

} // namespace cuda