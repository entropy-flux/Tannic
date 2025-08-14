#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic;

namespace cuda {  
    
status neg(const tensor_t*, tensor_t*, stream_t);
status add(const tensor_t*, const tensor_t*, tensor_t*, stream_t);
status sub(const tensor_t*, const tensor_t*, tensor_t*, stream_t); 
status mul(const tensor_t*, const tensor_t*, tensor_t*, stream_t);   
status pow(const tensor_t*, const tensor_t*, tensor_t*, stream_t);

} // namespace cuda