#pragma once
#include "core/types.h" 
#include "core/tensor.h"
#include <stdexcept>

namespace cpu {

void* allocate(size_t);    
void deallocate(void*);  

void copy(const void*, void*, size_t);
bool compare(const void*, const void*, size_t);
 
}