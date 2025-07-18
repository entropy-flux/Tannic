#include <stdexcept>
#include <cassert>
#include "Resources.hpp"  
#include <cstring>

using namespace tannic;  

void* Host::allocate(std::size_t nbytes) const {  
    return std::malloc(nbytes);
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    std::free(address);
}  

Device::Device(int id) : id_(id) {}
 
void* Device::allocate(std::size_t nbytes) const { 
    return nullptr;
}

void Device::deallocate(void* address, std::size_t nbytes) const {
} 