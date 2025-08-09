#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu { 
 
status log(tensor_t  const*, tensor_t*); 
status exp(tensor_t  const*, tensor_t*); 
status sqrt(tensor_t const*, tensor_t*); 
status rsqrt(tensor_t const*, tensor_t*, float); 
status abs(tensor_t  const*, tensor_t*); 
status sin(tensor_t  const*, tensor_t*); 
status cos(tensor_t  const*, tensor_t*); 
status tan(tensor_t  const*, tensor_t*); 
status sinh(tensor_t const*, tensor_t*); 
status cosh(tensor_t const*, tensor_t*); 
status tanh(tensor_t const*, tensor_t*);   

} // namespace cpu