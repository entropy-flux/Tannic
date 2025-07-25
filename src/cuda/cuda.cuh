#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/resources.h"

using namespace tannic;

namespace cuda { 
  
void checkError(cudaError_t err, const char* file, int line, const char* expr);

void* allocate(allocator_t const*, size_t);
void deallocate(allocator_t const*, void*, size_t);

#define CUDA_CHECK(call) checkError((call), __FILE__, __LINE__, #call)

} // namespace cuda 