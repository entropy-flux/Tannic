#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"
#include "runtime/streams.h"

using namespace tannic;

namespace cuda {

status repeat(const tensor_t*, tensor_t*, int, int, stream_t);

}