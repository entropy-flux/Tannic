#include <iostream>

#include "Operations.hpp"
#include "Tensor.hpp" 

template<typename T>
void add_kernel(const void* lhs, const void* rhs, void* out, size_t count) {
    const T* l = static_cast<const T*>(lhs);
    const T* r = static_cast<const T*>(rhs);
    T* o = static_cast<T*>(out);
    for (size_t i = 0; i < count; ++i) {
        o[i] = l[i] + r[i];
    }
}
 
void add_float32_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    const T* l = static_cast<const T*>(lhs);
    const T* r = static_cast<const T*>(rhs);
    T* o = static_cast<T*>(out);
    for (size_t i = 0; i < count; ++i) {
        o[i] = l[i] + r[i];
    }
}

void add_float64_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    add_kernel<double>(lhs, rhs, out, count);
}
 
template<class Operation>
using Kernels = void(*)(const void*, const void*, void*, size_t);
 
static Kernels<Addition> addition_kernels[TYPES][TYPES] = {
    [float32][float32] = add_float32_float32,
    [float64][float64] = add_float64_float64, 
};


void Negation::forward(Tensor const& operand, Tensor& result) {
    
}

void Addition::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) {
    std::cout << "Performing addition..." << std::endl;
}

void Subtraction::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) {
    std::cout << "Performing subtraction..." << std::endl;
}

void Multiplication::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) {
    std::cout << "Performing multiplication..." << std::endl;
} 