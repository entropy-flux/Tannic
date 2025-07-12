#include <cstdlib>  
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include "core/resources.h"
#include "cpu/cpu.hpp" 
#include "Types.hpp"

void* cpu::allocate(std::size_t nbytes) {  
    void* ptr = std::malloc(nbytes);
    if (ptr == nullptr) {
        throw std::runtime_error("CPU malloc failed for " + std::to_string(nbytes) + " bytes");
    }
    return ptr;
}

void cpu::deallocate(void* ptr) {
    if (ptr != nullptr) {  
        std::free(ptr);
    }
}

void cpu::copy(const void* src, void* dst, size_t nbytes) {
    if (src == nullptr || dst == nullptr) {
        throw std::invalid_argument("cpu::copy - source or destination is null");
    }
    std::memcpy(dst, src, nbytes);
}

bool cpu::compare(const void* a, const void* b, size_t nbytes) {
    if (a == nullptr || b == nullptr) {
        throw std::invalid_argument("cpu::compare - one or both inputs are null");
    }
    return std::memcmp(a, b, nbytes) == 0;
}