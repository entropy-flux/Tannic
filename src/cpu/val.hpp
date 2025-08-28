#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {

bool allclose(const tensor_t*, const tensor_t*,  double rtol = 1e-5, double atol = 1e-8);

}