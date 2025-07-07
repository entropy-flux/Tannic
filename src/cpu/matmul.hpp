#pragma once

#include <array>
#include "core/tensor.h"

namespace cpu {

template<typename TA, typename TB, typename TC>
void matmul_op(const tensor_t* A, const tensor_t* B, tensor_t* C, bool A_transposed, bool B_transposed);
  
} // namespace cpu


namespace cpu::matmul { 
 
using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

constexpr auto kernels = []() {
    
    std::array<std::array<Kernel, TYPES>, TYPES> table{}; 
    table[int8][int8]     = matmul_op<int8_t, int8_t, int32_t>;
    table[int8][int16]    = matmul_op<int8_t, int16_t, int32_t>;
    table[int8][int32]    = matmul_op<int8_t, int32_t, int32_t>;
    table[int8][int64]    = matmul_op<int8_t, int64_t, int64_t>;

    table[int16][int8]    = matmul_op<int16_t, int8_t, int32_t>;
    table[int16][int16]   = matmul_op<int16_t, int16_t, int32_t>;
    table[int16][int32]   = matmul_op<int16_t, int32_t, int32_t>;
    table[int16][int64]   = matmul_op<int16_t, int64_t, int64_t>;

    table[int32][int8]    = matmul_op<int32_t, int8_t, int32_t>;
    table[int32][int16]   = matmul_op<int32_t, int16_t, int32_t>;
    table[int32][int32]   = matmul_op<int32_t, int32_t, int64_t>;
    table[int32][int64]   = matmul_op<int32_t, int64_t, int64_t>;

    table[int64][int8]    = matmul_op<int64_t, int8_t, int64_t>;
    table[int64][int16]   = matmul_op<int64_t, int16_t, int64_t>;
    table[int64][int32]   = matmul_op<int64_t, int32_t, int64_t>;
    table[int64][int64]   = matmul_op<int64_t, int64_t, int64_t>;
 
    table[int32][float32] = matmul_op<int32_t, float, float>;
    table[float32][int32] = matmul_op<float, int32_t, float>;
    table[int32][float64] = matmul_op<int32_t, double, double>;
    table[float64][int32] = matmul_op<double, int32_t, double>;

    table[float32][float32] = matmul_op<float, float, float>;
    table[float32][float64] = matmul_op<float, double, double>;
    table[float64][float32] = matmul_op<double, float, double>;
    table[float64][float64] = matmul_op<double, double, double>;

    return table;
}();

} // namespace cpu::matmul