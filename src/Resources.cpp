#include <stdexcept>
#include "Resources.hpp" 
#include "cpu/cpu.hpp" 

#ifdef CUDA
#include "cuda/cuda.cuh"  

#include <cassert>
Host::Host() = default;

void* Host::allocate(std::size_t nbytes) const {  
    return cpu::allocate(nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    cpu::deallocate(address);
}

Device::Device(int id) : id_(id){}
 
void* Device::allocate(std::size_t nbytes) const { 
    return cuda::syncAllocate(nbytes, id_); 
}

void Device::deallocate(void* address, std::size_t nbytes) const {
    cuda::syncDeallocate(address, id_);
}
 
int Device::id() const {
    return id_;
}

#else 

Host::Host(bool pinned) : pinned_(pinned) {
    throw std::runtime_error("Pinned memory not supported without compute capabilities");
}


void* Host::allocate(std::size_t nbytes) const { 
    if (pinned_)  
        throw std::runtime_error("Pinned memory not supported without compute capabilities");
    else
        return cpu::allocate(nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {
    if (pinned_)
        throw std::runtime_error("Pinned memory not supported without compute capabilities");
    else
        cpu::deallocate(address);
} 

void* Device::allocate(std::size_t nbytes) const {  
    throw std::runtime_error("Device not supported");
}


void Device::deallocate(void* address, std::size_t nbytes) const {
    throw std::runtime_error("Device not supported");
}

Device::Device(int id) : id_(0){ 
    throw std::runtime_error("Device not supported");
}
 
#endif