#pragma once

#include <array>
#include "core/tensor.h"

namespace cpu {
  
template<typename TA, typename TB, typename TC, typename Op>
void binary_op(const tensor_t* A, const tensor_t* B, tensor_t* C, Op op); 

} // namespace cpu


namespace cpu::addition {

class Addition {
public:
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a + b)) -> decltype(a + b) {
        return a + b;
    }
};

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Addition);

constexpr auto kernels = []() {
    std::array<std::array<Kernel, TYPES>, TYPES> table{}; 
    
    table[int8][int8]   = binary_op<int8_t, int8_t, int8_t, Addition>;
    table[int8][int16]  = binary_op<int8_t, int16_t, int16_t, Addition>;
    table[int8][int32]  = binary_op<int8_t, int32_t, int32_t, Addition>;
    table[int8][int64]  = binary_op<int8_t, int64_t, int64_t, Addition>;

    table[int16][int8]  = binary_op<int16_t, int8_t, int16_t, Addition>;
    table[int16][int16] = binary_op<int16_t, int16_t, int16_t, Addition>;
    table[int16][int32] = binary_op<int16_t, int32_t, int32_t, Addition>;
    table[int16][int64] = binary_op<int16_t, int64_t, int64_t, Addition>;

    table[int32][int8]  = binary_op<int32_t, int8_t, int32_t, Addition>;
    table[int32][int16] = binary_op<int32_t, int16_t, int32_t, Addition>;
    table[int32][int32] = binary_op<int32_t, int32_t, int32_t, Addition>;
    table[int32][int64] = binary_op<int32_t, int64_t, int64_t, Addition>;

    table[int64][int8]  = binary_op<int64_t, int8_t, int64_t, Addition>;
    table[int64][int16] = binary_op<int64_t, int16_t, int64_t, Addition>;
    table[int64][int32] = binary_op<int64_t, int32_t, int64_t, Addition>;
    table[int64][int64] = binary_op<int64_t, int64_t, int64_t, Addition>;
 
    table[int32][float32] = binary_op<int32_t, float, float, Addition>;
    table[float32][int32] = binary_op<float, int32_t, float, Addition>;
    table[int32][float64] = binary_op<int32_t, double, double, Addition>;
    table[float64][int32] = binary_op<double, int32_t, double, Addition>;
 
    table[float32][float32] = binary_op<float, float, float, Addition>;
    table[float32][float64] = binary_op<float, double, double, Addition>;
    table[float64][float32] = binary_op<double, float, double, Addition>;
    table[float64][float64] = binary_op<double, double, double, Addition>;
    return table;
}();

} // namespace cpu::addition



namespace cpu::subtraction {

class Subtraction {
public:
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a - b)) -> decltype(a - b) {
        return a - b;
    }
};

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Subtraction);

constexpr auto kernels = []() {
    std::array<std::array<Kernel, TYPES>, TYPES> table{};

    table[int8][int8]   = binary_op<int8_t, int8_t, int8_t, Subtraction>;
    table[int8][int16]  = binary_op<int8_t, int16_t, int16_t, Subtraction>;
    table[int8][int32]  = binary_op<int8_t, int32_t, int32_t, Subtraction>;
    table[int8][int64]  = binary_op<int8_t, int64_t, int64_t, Subtraction>;

    table[int16][int8]  = binary_op<int16_t, int8_t, int16_t, Subtraction>;
    table[int16][int16] = binary_op<int16_t, int16_t, int16_t, Subtraction>;
    table[int16][int32] = binary_op<int16_t, int32_t, int32_t, Subtraction>;
    table[int16][int64] = binary_op<int16_t, int64_t, int64_t, Subtraction>;

    table[int32][int8]  = binary_op<int32_t, int8_t, int32_t, Subtraction>;
    table[int32][int16] = binary_op<int32_t, int16_t, int32_t, Subtraction>;
    table[int32][int32] = binary_op<int32_t, int32_t, int32_t, Subtraction>;
    table[int32][int64] = binary_op<int32_t, int64_t, int64_t, Subtraction>;

    table[int64][int8]  = binary_op<int64_t, int8_t, int64_t, Subtraction>;
    table[int64][int16] = binary_op<int64_t, int16_t, int64_t, Subtraction>;
    table[int64][int32] = binary_op<int64_t, int32_t, int64_t, Subtraction>;
    table[int64][int64] = binary_op<int64_t, int64_t, int64_t, Subtraction>;

    table[int32][float32] = binary_op<int32_t, float, float, Subtraction>;
    table[float32][int32] = binary_op<float, int32_t, float, Subtraction>;
    table[int32][float64] = binary_op<int32_t, double, double, Subtraction>;
    table[float64][int32] = binary_op<double, int32_t, double, Subtraction>;

    table[float32][float32] = binary_op<float, float, float, Subtraction>;
    table[float32][float64] = binary_op<float, double, double, Subtraction>;
    table[float64][float32] = binary_op<double, float, double, Subtraction>;
    table[float64][float64] = binary_op<double, double, double, Subtraction>;

    return table;
}();

} // namespace cpu::subtraction




namespace cpu::multiplication {

class Multiplication {
public:
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a * b)) -> decltype(a * b) {
        return a * b;
    }
};

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Multiplication);

constexpr auto kernels = []() {
    std::array<std::array<Kernel, TYPES>, TYPES> table{};

    table[int8][int8]   = binary_op<int8_t, int8_t, int8_t, Multiplication>;
    table[int8][int16]  = binary_op<int8_t, int16_t, int16_t, Multiplication>;
    table[int8][int32]  = binary_op<int8_t, int32_t, int32_t, Multiplication>;
    table[int8][int64]  = binary_op<int8_t, int64_t, int64_t, Multiplication>;

    table[int16][int8]  = binary_op<int16_t, int8_t, int16_t, Multiplication>;
    table[int16][int16] = binary_op<int16_t, int16_t, int16_t, Multiplication>;
    table[int16][int32] = binary_op<int16_t, int32_t, int32_t, Multiplication>;
    table[int16][int64] = binary_op<int16_t, int64_t, int64_t, Multiplication>;

    table[int32][int8]  = binary_op<int32_t, int8_t, int32_t, Multiplication>;
    table[int32][int16] = binary_op<int32_t, int16_t, int32_t, Multiplication>;
    table[int32][int32] = binary_op<int32_t, int32_t, int32_t, Multiplication>;
    table[int32][int64] = binary_op<int32_t, int64_t, int64_t, Multiplication>;

    table[int64][int8]  = binary_op<int64_t, int8_t, int64_t, Multiplication>;
    table[int64][int16] = binary_op<int64_t, int16_t, int64_t, Multiplication>;
    table[int64][int32] = binary_op<int64_t, int32_t, int64_t, Multiplication>;
    table[int64][int64] = binary_op<int64_t, int64_t, int64_t, Multiplication>;

    table[int32][float32] = binary_op<int32_t, float, float, Multiplication>;
    table[float32][int32] = binary_op<float, int32_t, float, Multiplication>;
    table[int32][float64] = binary_op<int32_t, double, double, Multiplication>;
    table[float64][int32] = binary_op<double, int32_t, double, Multiplication>;

    table[float32][float32] = binary_op<float, float, float, Multiplication>;
    table[float32][float64] = binary_op<float, double, double, Multiplication>;
    table[float64][float32] = binary_op<double, float, double, Multiplication>;
    table[float64][float64] = binary_op<double, double, double, Multiplication>;

    return table;
}();

} // namespace cpu::multiplication
 