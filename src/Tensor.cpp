#include "cpu/mem.hpp"
#include "cuda/mem.cuh"
#include "Tensor.hpp"   

using namespace tannic;

void Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {      
    std::byte* target = static_cast<std::byte*>(storage_->address()) + offset;  
    if (resource() == HOST) {
        cpu::copy(value, target, dsizeof(dtype_));
    } 
    
    else {
        throw std::runtime_error("copy not implemented for this resource");
    }
} 

bool Tensor::compare(std::byte const* value, std::ptrdiff_t offset) const {  
    std::byte const* target = static_cast<std::byte const*>(storage_->address()) + offset; 
    if (resource() == HOST) {
        return cpu::compare(value, target, dsizeof(dtype_));
    } 
    
    else {
        throw std::runtime_error("copy not implemented for this resource");
    }
}