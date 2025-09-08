#include "runtime/types.h"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/status.h" 

using namespace tannic;

namespace cpu {

status cpy(const tensor_t* src, tensor_t* dst);

}