#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {

status triu(const tensor_t* src, tensor_t* dst, int k = 0);
status tril(const tensor_t* src, tensor_t* dst, int k = 0);

}