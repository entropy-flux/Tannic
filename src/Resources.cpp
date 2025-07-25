#include <stdexcept>
#include <cassert> 
#include <cstring>  
#include "Resources.hpp"
#include "Bindings.hpp" 
#include "cuda/mem.cuh"

namespace tannic { 

#ifdef CUDA   

void* Host::allocate(std::size_t nbytes) const {    
    host_t resource = structure(*this);
    if(resource.traits & PAGEABLE)  
        return std::malloc(nbytes);
    else 
        return cuda::allocate(&resource, nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    host_t resource = structure(*this);
    if(resource.traits & PAGEABLE)  
        std::free(address);
    else 
        return cuda::deallocate(&resource, address, nbytes);
}  

Devices::Devices() {
    count_ = cuda::getDeviceCount();
}

Device::Device(int id) : id_(id) {
    if (id < 0 || id >= Devices::count()) {
        throw std::out_of_range("Invalid device ID"); 
    }
}
 
void* Device::allocate(std::size_t nbytes) const { 
    throw std::runtime_error("Device not supported without compute capabilities");
}

void Device::deallocate(void* address, std::size_t nbytes) const {
    throw std::runtime_error("Device not supported without compute capabilities");
} 

#elif
 

void* Host::allocate(std::size_t nbytes) const {  
    return std::malloc(nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    std::free(address);
}  

Device::Device(int id) : id_(id) {
    throw std::runtime_error("Device not supported without compute capabilities");
}
 
void* Device::allocate(std::size_t nbytes) const { 
    throw std::runtime_error("Device not supported without compute capabilities");
}

void Device::deallocate(void* address, std::size_t nbytes) const {
    throw std::runtime_error("Device not supported without compute capabilities");
} 

#endif

} // namespace tannic