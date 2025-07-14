#pragma once 
#include <stdexcept>

namespace cpu {

void* allocate(std::size_t);    
void deallocate(std::byte*);  

void copy(std::byte const*, std::byte*, std::size_t);
bool compare(std::byte const*, std::byte const*, std::size_t);
 
}