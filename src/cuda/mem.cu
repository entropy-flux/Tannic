#include <sstream>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "cuda/cuda.cuh"

#include <sstream>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "cuda/cuda.cuh"
#include "core/resources.h"

void* cuda::syncHostAllocate(std::size_t nbytes) { 
    void* address = nullptr;
    CUDA_CHECK(cudaHostAlloc(&address,nbytes, cudaHostAllocDefault));
    return address;
}

void cuda::syncHostDeallocate(void* address) {
    if (address != nullptr) { 
        CUDA_CHECK(cudaFreeHost(address));
    }
}

void* cuda::syncAllocate(std::size_t size, int device) { 
    void* address = nullptr;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&address, size)); 
    return address;
}

void cuda::syncDeallocate(void* address, int device) {
    if (address != nullptr) { 
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(address));
    }
}