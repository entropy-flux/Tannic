#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h" 

using namespace tannic;

namespace cpu {  
      
status view_as_cartesian(tensor_t const*, tensor_t const*, tensor_t*);   
status view_as_polar(const tensor_t*, const tensor_t*, tensor_t*);

} // namespace cpu