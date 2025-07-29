#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cpu {  
    
void neg(tensor_t const*, tensor_t*);
void add(tensor_t const*, tensor_t const*, tensor_t*);
void sub(tensor_t const*, tensor_t const*, tensor_t*); 
void mul(tensor_t const*, tensor_t const*, tensor_t*);    

} // namespace cpu