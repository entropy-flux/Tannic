#include "Memory/Resources.hpp"
#include <iostream>

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