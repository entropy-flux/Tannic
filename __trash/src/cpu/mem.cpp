#include <cstdlib>  
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector> 
#include "cpu/mem.hpp"  

void* cpu::allocate(std::size_t nbytes) {  
    void* ptr = std::malloc(nbytes);
    if (ptr == nullptr) {
        throw std::runtime_error("CPU malloc failed for " + std::to_string(nbytes) + " bytes");
    }
    return ptr;
}

void cpu::deallocate(std::byte* ptr) {
    if (ptr != nullptr) {  
        std::free(ptr);
    }
}

void cpu::copy(std::byte const* src, std::byte* dst, std::size_t nbytes) {
    if (src == nullptr || dst == nullptr) {
        throw std::invalid_argument("cpu::copy - source or destination is null");
    }
    std::memcpy(dst, src, nbytes);
}

bool cpu::compare(std::byte const* a, std::byte const* b, std::size_t nbytes) {
    if (a == nullptr || b == nullptr) {
        throw std::invalid_argument("cpu::compare - one or both inputs are null");
    }
    return std::memcmp(a, b, nbytes) == 0;
}