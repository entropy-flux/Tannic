#pragma once
#include "core/types.h" 
#include <stdexcept>

namespace cpu {
   
constexpr inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES)*static_cast<int>(second);
} // todo remove from here. 

void* allocate(size_t);    
void deallocate(void*);  
 
}