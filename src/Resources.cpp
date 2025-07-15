#include <stdexcept>
#include <cassert>
#include "Resources.hpp" 
#include "cpu/mem.hpp"  
#include "cuda/mem.cuh"    

using namespace tannic;  

void* Host::allocate(std::size_t nbytes) const {  
    return cpu::allocate(nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    cpu::deallocate(address);
}  

Device::Device(int id) : id_(id) {}
 
void* Device::allocate(std::size_t nbytes) const { 
    return nullptr;
}

void Device::deallocate(void* address, std::size_t nbytes) const {
} 