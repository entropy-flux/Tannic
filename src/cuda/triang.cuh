#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"
#include "runtime/streams.h"

using namespace tannic;

namespace cuda {

status triu(const tensor_t* src, tensor_t* dst, stream_t stream, int k = 0);
status tril(const tensor_t* src, tensor_t* dst, stream_t stream, int k = 0);

}