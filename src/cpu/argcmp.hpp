#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cpu {  
     
void argmax(tensor_t const*, tensor_t*, uint8_t);
void argmin(tensor_t const*, tensor_t*, uint8_t);

} // namespace cpu