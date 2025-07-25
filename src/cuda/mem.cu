#include <stdexcept>
#include "cuda/mem.cuh"
#include "cuda/exc.cuh"

namespace cuda {   

int getDeviceCount() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count); CUDA_CHECK(err);
    return count;
}

void* allocate(host_t const* resource, size_t nbytes) { 
    cudaError_t err;  
    if (resource->traits & PINNED) {
        void* ptr = nullptr; 
        err = cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault); CUDA_CHECK(err);
        return ptr;
    }
    else {
        throw std::runtime_error("Allocation not supported by CUDA allocator.");
    }      
}

void deallocate(host_t const* resource, void* address, size_t nbytes) {  
    cudaError_t err;   
    if (resource->traits & PINNED) {
        err = cudaFreeHost(address); CUDA_CHECK(err);
    }

    else {
        throw std::runtime_error("Dellocation not supported by CUDA allocator.");
    }
 
}

void* allocate(device_t const* resource, size_t nbytes) { 
    cudaError_t err;  
    void* ptr = nullptr;
    err = cudaSetDevice(resource->id);  CUDA_CHECK(err);
    err = cudaMalloc(&ptr, nbytes); CUDA_CHECK(err); 
    return ptr; 
}

void deallocate(device_t const* resource, void* address, size_t nbytes) {  
    cudaError_t err;  
    err = cudaSetDevice(resource->id); CUDA_CHECK(err);
    err = cudaFree(address); CUDA_CHECK(err); 
}

} // namespace cuda 