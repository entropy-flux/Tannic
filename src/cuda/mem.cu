#include "cuda/cuda.cuh"

namespace cuda {   

void* allocate(allocator_t const* allocator, size_t nbytes) { 
    cudaError_t err; 
    if (allocator->environment == HOST) {
        void* ptr = nullptr; 
        err = cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault); CUDA_CHECK(err);
        return ptr;
    } else {
        void* ptr = nullptr;
        err = cudaSetDevice(allocator->resource.device.id);  CUDA_CHECK(err);
        err = cudaMalloc(&ptr, nbytes); CUDA_CHECK(err); 
        return ptr;
    }
}

void deallocate(allocator_t const* allocator, void* address, size_t nbytes) {  
    if (allocator->environment == HOST) {
        cudaFreeHost(address);   
    } else {
        cudaSetDevice(allocator->resource.device.id);
        cudaFree(address);
    }
}

} // namespace cuda 