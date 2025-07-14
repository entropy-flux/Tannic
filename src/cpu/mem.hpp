#pragma once 
#include <stdexcept>

namespace cpu { 
void* allocate(std::size_t);    
void deallocate(void*);   
void copy(const std::byte*, std::byte*, std::size_t);
bool compare(const std::byte*, const std::byte*, std::size_t); 
}