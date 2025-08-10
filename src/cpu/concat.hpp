#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu { 
 
status concat(tensor_t const*, tensor_t const*, tensor_t*, int dim);  

} // namespace cpu