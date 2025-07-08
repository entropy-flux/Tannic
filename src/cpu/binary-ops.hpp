#pragma once

#include "core/tensor.h"
#include "cpu/cpu.hpp"
#include <array> 

namespace {
    
[[noreturn]] inline void notImplemented(const tensor_t*, const tensor_t*, tensor_t*, auto) {
    throw std::runtime_error("Kernel not implemented for this type");
} 

}

namespace cpu { 

template<typename S, typename D, typename TC, typename Op>
void binaryOp(const tensor_t* A, const tensor_t* B, tensor_t* C, Op op); 

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
    std::array<Kernel, TYPES * TYPES> table;
    table.fill(::notImplemented);

    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Addition>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Addition>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Addition>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Addition>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Addition>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Addition>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Addition>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Addition>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Addition>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Addition>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Addition>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Addition>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Addition>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Addition>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Addition>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Addition>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Addition>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Addition>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Addition>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Addition>;

    table[index(float32, float32)] = binaryOp<float, float, float, Addition>;
    table[index(float32, float64)] = binaryOp<float, double, double, Addition>;
    table[index(float64, float32)] = binaryOp<double, float, double, Addition>;
    table[index(float64, float64)] = binaryOp<double, double, double, Addition>;

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
    std::array<Kernel, TYPES * TYPES> table;
    table.fill(::notImplemented);

    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Subtraction>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Subtraction>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Subtraction>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Subtraction>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Subtraction>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Subtraction>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Subtraction>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Subtraction>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Subtraction>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Subtraction>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Subtraction>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Subtraction>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Subtraction>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Subtraction>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Subtraction>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Subtraction>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Subtraction>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Subtraction>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Subtraction>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Subtraction>;

    table[index(float32, float32)] = binaryOp<float, float, float, Subtraction>;
    table[index(float32, float64)] = binaryOp<float, double, double, Subtraction>;
    table[index(float64, float32)] = binaryOp<double, float, double, Subtraction>;
    table[index(float64, float64)] = binaryOp<double, double, double, Subtraction>;

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
    std::array<Kernel, TYPES * TYPES> table;
    table.fill(::notImplemented);

    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Multiplication>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Multiplication>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Multiplication>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Multiplication>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Multiplication>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Multiplication>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Multiplication>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Multiplication>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Multiplication>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Multiplication>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Multiplication>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Multiplication>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Multiplication>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Multiplication>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Multiplication>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Multiplication>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Multiplication>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Multiplication>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Multiplication>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Multiplication>;

    table[index(float32, float32)] = binaryOp<float, float, float, Multiplication>;
    table[index(float32, float64)] = binaryOp<float, double, double, Multiplication>;
    table[index(float64, float32)] = binaryOp<double, float, double, Multiplication>;
    table[index(float64, float64)] = binaryOp<double, double, double, Multiplication>;

    return table;
}();

} // namespace cpu::multiplication