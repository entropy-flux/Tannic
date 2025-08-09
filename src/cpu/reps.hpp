#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {

status repeat(const tensor_t*, tensor_t*, int, int);

}