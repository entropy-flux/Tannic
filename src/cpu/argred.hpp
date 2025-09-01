#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {  
     
status argmax(tensor_t const*, tensor_t*, uint8_t);
status argmin(tensor_t const*, tensor_t*, uint8_t);
status argsum(tensor_t const*, tensor_t*, uint8_t);
status argmean(tensor_t const*, tensor_t*, uint8_t);

} // namespace cpu