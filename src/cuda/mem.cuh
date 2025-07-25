#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/resources.h"
#include <cuda_runtime.h>

using namespace tannic;

namespace cuda { 
  
int getDeviceCount(); 
void* allocate(host_t const*, size_t);
void* allocate(device_t const*, size_t);
void deallocate(host_t const*, void*, size_t);
void deallocate(device_t const*, void*, size_t);  

} // namespace cuda 