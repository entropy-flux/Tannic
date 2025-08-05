#pragma once
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic; 

namespace cuda {
 
status log( tensor_t const*, tensor_t*, stream_t);
status exp( tensor_t const*, tensor_t*, stream_t);
status sqrt(tensor_t const*, tensor_t*, stream_t);
status abs( tensor_t const*, tensor_t*, stream_t);
status sin( tensor_t const*, tensor_t*, stream_t);
status cos( tensor_t const*, tensor_t*, stream_t);
status tan( tensor_t const*, tensor_t*, stream_t);
status sinh(tensor_t const*, tensor_t*, stream_t);
status cosh(tensor_t const*, tensor_t*, stream_t);
status tanh(tensor_t const*, tensor_t*, stream_t);

}