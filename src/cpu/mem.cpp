#include <cstdlib>  
#include <stdexcept>
#include <string>
#include <cstring>
#include "core/resources.h"
#include "cpu/cpu.hpp"

void* cpu::allocate(std::size_t nbytes) {  
    void* address = std::malloc(nbytes);
    if (address == nullptr) {
        throw std::runtime_error("CPU malloc failed for " + std::to_string(nbytes) + " bytes");
    }
    return address;
}

void cpu::deallocate(void* address) {
    if (address != nullptr) {  
        std::free(address);
    }
}