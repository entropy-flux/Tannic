#pragma once
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"

using namespace tannic; 

namespace cuda {
 
void log( tensor_t const*, tensor_t*, stream_t);
void exp( tensor_t const*, tensor_t*, stream_t);
void sqrt(tensor_t const*, tensor_t*, stream_t);
void abs( tensor_t const*, tensor_t*, stream_t);
void sin( tensor_t const*, tensor_t*, stream_t);
void cos( tensor_t const*, tensor_t*, stream_t);
void tan( tensor_t const*, tensor_t*, stream_t);
void sinh(tensor_t const*, tensor_t*, stream_t);
void cosh(tensor_t const*, tensor_t*, stream_t);
void tanh(tensor_t const*, tensor_t*, stream_t);

}