#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"
#include "runtime/streams.h"

using namespace tannic;

namespace cuda {

bool allclose(const tensor_t*, const tensor_t*, stream_t, double rtol = 1e-5, double atol = 1e-8);

}