#pragma once  
#include "runtime/resources.h"
#include "runtime/tensor.h"
#include "runtime/streams.h"

using namespace tannic;

namespace cuda { 

int getDeviceCount();
void setDevice(int /*device id*/);
void* allocate(const device_t*, size_t /*number of bytes*/);
void* deallocate(const device_t*, void*  /*memory address*/); 
void copyFromHost(const device_t*, const void* /*host memory address*/, void* /*device memory address*/, size_t /*number of bytes*/);  
bool compareFromHost(const device_t*, const void* /*host memory address*/, const void* /*device memory address*/, size_t /*number of bytes*/);
void copyDeviceToHost(const device_t*, const void*, void*, size_t);

status copy(const tensor_t* src, tensor_t* dst, size_t nbytes, stream_t stream);

} // namespace cuda 