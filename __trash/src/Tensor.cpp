#include "core/tensor.h"
#include "cpu/cpu.hpp"
#include "cuda/cuda.cuh"
#include "Tensor.hpp"  

void Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {     
    assert(is_initialized() && "Cannot assign a value to an uninitialized tensor.");
    auto copy = [this](std::byte const* source, std::byte* target, std::size_t size) {
        if (resource() == HOST) {
            cpu::copy(source, target, size);
        } 
        
        else {
            throw std::runtime_error("copy not implemented for this resource");
        }
    }; 
    
    std::size_t dsize = dsizeof(dtype_); 
    std::byte* target = address() + offset;

    if (rank() == 0) {
        copy(value, target, dsize);
        return;
    } 
  
    std::vector<size_t> indices(rank(), 0);

    for (std::size_t i = 0; i < shape_.size(); ++i) {
        std::ptrdiff_t position = 0;

        for (rank_type dimension = 0; dimension < rank(); ++dimension) {
            position += indices[dimension] * strides_[dimension];
        }
        copy(value, target + position * dsize, dsize);

        for (int dimension = rank() - 1; dimension >= 0; --dimension) {
            if (++indices[dimension] < shape_[dimension]) {
                break;
            }
            indices[dimension] = 0;
        }
    } 
}

bool Tensor::compare(std::byte const* value, std::ptrdiff_t offset) const { 
    assert(is_initialized() && "Cannot compare a value to an uninitialized tensor."); 
    auto compare = [this](std::byte const* source, std::byte const* target, std::size_t size) -> bool {
        if (resource() == HOST) {
            return cpu::compare(source, target, size);
        } 
        
        else {
            throw std::runtime_error("copy not implemented for this resource");
        }
    }; 
    std::byte const* target = address() + offset; 
    return compare(value, target, dsizeof(dtype_));
}