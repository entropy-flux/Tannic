#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {   

status eq(const tensor_t*, const tensor_t*, tensor_t*);
status ne(const tensor_t*, const tensor_t*, tensor_t*);
status gt(const tensor_t*, const tensor_t*, tensor_t*);
status ge(const tensor_t*, const tensor_t*, tensor_t*);
status lt(const tensor_t*, const tensor_t*, tensor_t*);
status le(const tensor_t*, const tensor_t*, tensor_t*);

} // namespace cpu