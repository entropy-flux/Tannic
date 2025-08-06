#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/streams.h"
#include "runtime/status.h" 

using namespace tannic;

namespace cuda {  
     
status argmax(tensor_t const*, tensor_t*, uint8_t, stream_t);
status argmin(tensor_t const*, tensor_t*, uint8_t, stream_t);

} // namespace cuda