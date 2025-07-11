#pragma once
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h> 
#include "core/types.h"

namespace cuda {

constexpr inline auto index(type dtype) { return static_cast<int>(dtype); }
constexpr inline auto index(type first, type second) { return static_cast<int>(first) + static_cast<int>(TYPES)*static_cast<int>(second); }
  
void checkError(cudaError_t status, const char* message, const char* file, int line);
void* syncHostAllocate(std::size_t nbytes);
void syncHostDeallocate(void* ptr);
void* syncAllocate(std::size_t size, int device);
void syncDeallocate(void* ptr, int device); 

} // namespace cuda

#define CUDA_CHECK(call) cuda::checkError((call), #call, __FILE__, __LINE__)