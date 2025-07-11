#pragma once

#include <array> 
#include "core/tensor.h"
#include "cpu/cpu.hpp"
 

namespace {

[[noreturn]] inline void notImplemented(const tensor_t*, const tensor_t*, tensor_t*, bool, bool) {
    throw std::runtime_error("Kernel not implemented for this type");
} 

}

namespace cpu {

template<typename S, typename D, typename TC>
void matmulOp(const tensor_t* A, const tensor_t* B, tensor_t* C, bool A_transposed, bool B_transposed);
 
} // namespace cpu
 

namespace cpu::matmul { 
 
using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES * TYPES> table;
    table.fill(::notImplemented);

    table[index(int8, int8)]     = matmulOp<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = matmulOp<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = matmulOp<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = matmulOp<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = matmulOp<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = matmulOp<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = matmulOp<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = matmulOp<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = matmulOp<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = matmulOp<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = matmulOp<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = matmulOp<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = matmulOp<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = matmulOp<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = matmulOp<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = matmulOp<int64_t, int64_t, int64_t>;

    table[index(int32, float32)] = matmulOp<int32_t, float, float>;
    table[index(float32, int32)] = matmulOp<float, int32_t, float>;
    table[index(int32, float64)] = matmulOp<int32_t, double, double>;
    table[index(float64, int32)] = matmulOp<double, int32_t, double>;

    table[index(float32, float32)] = matmulOp<float, float, float>;
    table[index(float32, float64)] = matmulOp<float, double, double>;
    table[index(float64, float32)] = matmulOp<double, float, double>;
    table[index(float64, float64)] = matmulOp<double, double, double>;

    return table;
}();

}