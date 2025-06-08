#include "include/Tensor.hpp"
#include <fstream>
#include <cstddef>


unsigned long long Host::available() const {
    #ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open()) {
            throw std::runtime_error("Failed to open /proc/meminfo");
        }

        std::string line;
        unsigned long long available_kb = 0;

        while (std::getline(meminfo, line)) {
            if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %llu kB", &available_kb);
                break;
            }
        }

        if (available_kb == 0) {
            throw std::runtime_error("MemAvailable not found in /proc/meminfo");
        }

        return available_kb * 1024ULL;  // Return bytes
    #else
        throw std::runtime_error("Unsupported OS for Memory::available()");
    #endif
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

    if (!address) {
        std::cerr << "Returning a nullptr" << "\n";
    }

    return address;
}

void Device::deallocate(void* address, std::size_t) {
    cudaSetDevice(id);
    cudaFree(address);
}




int main() {
    Tensor x(float32, {2, 3}, Device(0)); 
    void* address = x.address();  
    float value = 5;
    cudaMemcpy(address, &value, sizeof(float), cudaMemcpyHostToDevice);
    
}