#include <iostream>
#include <string> 
#include <vector>
#include <cuda_runtime.h>
 
#include <cstddef>
#include <vector>
#include <span>

enum unit : std::size_t {
    B = 1,
    KB = 1024,
    MB = 1024 * 1024,
    GB = 1024 * 1024 * 1024,
    UNIT
};

struct Device {
    int id;

    void* allocate(std::size_t memory);               
    void deallocate(void* address, std::size_t size); 
    unsigned long long available() const;
};

struct Host {
public:
    void* allocate(std::size_t memory) const { return ::operator new(memory); }
    void deallocate(void* address, std::size_t) const { ::operator delete(address); }
    unsigned long long available() const;
};

class Resources {
public:
    Resources();
    static Host host() { return Host{}; } 
    std::span<const Device> devices() const {
        return std::span<const Device>(devices_.data(), devices_.size());
    }

private:
    std::vector<Device> devices_;
};

Resources::Resources() {
    try {
        int count;    
        cudaGetDeviceCount(&count);
        for(int id = 0; id < count; id++) {
            devices_.emplace_back(Device{id});
        }
    } 
    
    catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl; 
    }
}
 

// Implement Device member functions
void* Device::allocate(std::size_t memory) {
    void* ptr = nullptr;
    cudaError_t err = cudaSetDevice(id);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDevice failed for device " << id << ": " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    err = cudaMalloc(&ptr, memory);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for device " << id << ": " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    return ptr;
}

void Device::deallocate(void* address, std::size_t) {
    cudaSetDevice(id);
    cudaFree(address);
}

unsigned long long Device::available() const {
    cudaSetDevice(id);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return static_cast<unsigned long long>(free_mem);
}

unsigned long long Host::available() const {
    // This is platform dependent; 
    // here we return a dummy value or implement a heuristic if you want
    return 16ULL * unit::GB; // Assume 16GB host RAM available (just an example)
}

int main() {
    Resources resources;

    std::cout << "Host available memory: " << resources.host().available() / unit::MB << " MB\n";

    auto devices = resources.devices();
    std::cout << "Found " << devices.size() << " CUDA devices.\n";

    for (const auto& device : devices) {
        std::cout << "Device ID: " << device.id 
                  << ", Available memory: " << device.available() / unit::MB << " MB\n";
    }

    return 0;
}