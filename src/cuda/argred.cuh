#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/streams.h"
#include "runtime/status.h" 

using namespace tannic;

namespace cuda {  
     
status argsum(const tensor_t*, tensor_t*, stream_t, uint8_t);
status argmean(const tensor_t*, tensor_t*, stream_t, uint8_t);

} // namespace cuda