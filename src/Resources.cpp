#include <stdexcept>
#include <cassert> 
#include <cstring> 
#include "runtime/resources.h"
#include "Resources.hpp"

namespace tannic { 

#ifdef CUDA   

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