#include "Tensor.hpp"   
#include <cstring>

using namespace tannic;

void Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {      
    std::byte* target = static_cast<std::byte*>(storage_->address()) + offset;  
    if (environment() == HOST) {
        std::memcpy(target, value, dsizeof(dtype_));
    } 
    
    else {
        throw std::runtime_error("copy not implemented for this resource");
    }
} 

bool Tensor::compare(std::byte const* value, std::ptrdiff_t offset) const {  
    std::byte const* target = static_cast<std::byte const*>(storage_->address()) + offset; 
    if (environment() == HOST) {
        return std::memcmp(value, target, dsizeof(dtype_)) == 0;
    } 
    
    else {
        throw std::runtime_error("copy not implemented for this resource");
    }
}