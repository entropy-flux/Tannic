#include "Resources.hpp"
#include <iostream>
 
void* Device::allocate(std::size_t memory) {
    void* address = nullptr;
    cudaError_t error = cudaSetDevice(id);
    if (error != cudaSuccess) {
        std::cerr << "cudaSetDevice failed for device " << id << ": " << cudaGetErrorString(error) << "\n";
        return nullptr;
    }
    error = cudaMalloc(&address, memory);
    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed for device " << id << ": " << cudaGetErrorString(error) << "\n";
        return nullptr;
    }
    return address;
}

void Device::deallocate(void* address, std::size_t) {
    cudaSetDevice(id);
    cudaFree(address);
}

void Device::copy(void* address, void const* value, std::size_t size, Processor processor) const {
    cudaMemcpy(address, value, size, cudaMemcpyHostToDevice);
}

bool Device::compare(void const* address, void const* value, std::size_t size, Processor processor) const {
    void* buffer = std::malloc(size);
    if (!buffer) {
        throw std::bad_alloc();
    }

    cudaError_t status = cudaMemcpy(buffer, address, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::free(buffer);
        throw std::runtime_error("cudaMemcpy failed during comparision: " + std::string(cudaGetErrorString(status)));
    }

    bool result = std::memcmp(buffer, value, size) == 0;
    std::free(buffer);
    return result;
}

Resources::Resources() {
    try {
        int count;    
        cudaGetDeviceCount(&count);
        for(int id = 0; id < count; id++) {
            devices_.emplace_back(id);
        }
    } 
    
    catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl; 
    }
}