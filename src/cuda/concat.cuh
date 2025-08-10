#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"
#include "runtime/streams.h"

using namespace tannic;

namespace cuda { 

status concat(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int);

} // namespace cuda