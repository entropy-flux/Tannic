#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic;

namespace cuda {  
    
status neg(tensor_t const*, tensor_t*, stream_t);
status add(tensor_t const*, tensor_t const*, tensor_t*, stream_t);
status sub(tensor_t const*, tensor_t const*, tensor_t*, stream_t); 
status mul(tensor_t const*, tensor_t const*, tensor_t*, stream_t);    

} // namespace cuda