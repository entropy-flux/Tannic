#include <cstdlib>  
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector> 
#include "cpu/mem.hpp"  

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

void cpu::copy(const std::byte* source, std::byte* target, std::size_t nbytes) {
    if (source == nullptr || target == nullptr) {
        throw std::invalid_argument("cpu::copy - source or destination is null");
    }
    std::memcpy(static_cast<void*>(target), static_cast<const void*>(source), nbytes);
}

bool cpu::compare(const std::byte* first, const std::byte* second, std::size_t nbytes) {
    if (first == nullptr || second == nullptr) {
        throw std::invalid_argument("cpu::compare - one or both inputs are null");
    }
    return std::memcmp(static_cast<const void*>(first), static_cast<const void*>(second), nbytes) == 0;
}