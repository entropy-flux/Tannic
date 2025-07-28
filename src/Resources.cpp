#include <stdexcept>
#include <cassert> 
#include <cstring>  
#include "Resources.hpp"
#include "Bindings.hpp" 
#include "cuda/mem.cuh"

#ifdef CUDA   

namespace tannic { 

void* Host::allocate(std::size_t nbytes) const {     
    if(pageable_) {
        return std::malloc(nbytes);
    } else {
        throw std::runtime_error("ERROR!");
    }
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    if(pageable_) {
        std::free(address); 
    } else {
        throw std::runtime_error("ERROR!");
    }
}

Devices::Devices() {
    count_ = cuda::getDeviceCount();
}

Device::Device(int id, bool blocking) 
:   id_(id)
,   blocking_(blocking) { 
    if (id < 0 || id >= Devices::count()) {
        throw std::out_of_range("Invalid device ID"); 
    } 
}

void* Device::allocate(std::size_t nbytes) const {  
    device_t resource = structure(*this); 
    return cuda::allocate(&resource, nbytes); 
}

void Device::deallocate(void* address, std::size_t nbytes) const {
    device_t resource = structure(*this);  
    cuda::deallocate(&resource, address); 
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