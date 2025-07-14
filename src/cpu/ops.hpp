#pragma once
#include <array>
#include <cmath>  
#include <stdexcept>
#include "ctypes/types.h"
#include "ctypes/tensor.h" 

namespace cpu {

enum class Operation {
    NEG,
    ADD,
    SUB,
    MUL
};

namespace unary {

static constexpr inline int index(type type) {
    return static_cast<int>(type);
}

using Kernel = bool(*)(const tensor_t*, tensor_t*);    

template<typename S, typename T, Operation O>
bool operation(const tensor_t*, tensor_t*); 
extern template bool operation<int8_t, int8_t, Operation::NEG>(const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int16_t, Operation::NEG>(const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int32_t, Operation::NEG>(const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int64_t, Operation::NEG>(const tensor_t*, tensor_t*);
extern template bool operation<float, float, Operation::NEG>(const tensor_t*, tensor_t*);
extern template bool operation<double, double, Operation::NEG>(const tensor_t*, tensor_t*);

constexpr auto negation = []() {  
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(int8)] = operation<int8_t, int8_t, Operation::NEG>;
    table[index(int16)] = operation<int16_t, int16_t, Operation::NEG>;
    table[index(int32)] = operation<int32_t, int32_t, Operation::NEG>;
    table[index(int64)] = operation<int64_t, int64_t, Operation::NEG>;
    table[index(float32)] = operation<float, float, Operation::NEG>;
    table[index(float64)] = operation<double, double, Operation::NEG>;
    return table;
}();

} // namespace unary


namespace binary {

static constexpr inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
} 

using Kernel = bool(*)(const tensor_t*, const tensor_t*, tensor_t*);    

template<typename S1, typename S2, typename T, Operation F>
bool operation(const tensor_t*, const tensor_t*, tensor_t*); 

extern template bool operation<int8_t, int8_t, int8_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int16_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int8_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int16_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int8_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int16_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int8_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int16_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int32_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, float, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, int32_t, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, int32_t, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, float, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, float, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
  
constexpr auto addition = []() {
    std::array<Kernel, index(TYPES, TYPES)> table = {[](const tensor_t*, const tensor_t*, tensor_t*) { return false; }}; 
    table[index(int8, int8)]   = operation<int8_t, int8_t, int8_t, Operation::ADD>;
    table[index(int8, int16)]  = operation<int8_t, int16_t, int16_t, Operation::ADD>; 
    table[index(int8, int32)]  = operation<int8_t, int32_t, int32_t, Operation::ADD>;
    table[index(int8, int64)]  = operation<int8_t, int64_t, int64_t, Operation::ADD>;

    table[index(int16, int8)]  = operation<int16_t, int8_t, int16_t, Operation::ADD>;
    table[index(int16, int16)] = operation<int16_t, int16_t, int16_t, Operation::ADD>;
    table[index(int16, int32)] = operation<int16_t, int32_t, int32_t, Operation::ADD>;
    table[index(int16, int64)] = operation<int16_t, int64_t, int64_t, Operation::ADD>;

    table[index(int32, int8)]  = operation<int32_t, int8_t, int32_t, Operation::ADD>;
    table[index(int32, int16)] = operation<int32_t, int16_t, int32_t, Operation::ADD>;
    table[index(int32, int32)] = operation<int32_t, int32_t, int32_t, Operation::ADD>;
    table[index(int32, int64)] = operation<int32_t, int64_t, int64_t, Operation::ADD>;

    table[index(int64, int8)]  = operation<int64_t, int8_t, int64_t, Operation::ADD>;
    table[index(int64, int16)] = operation<int64_t, int16_t, int64_t, Operation::ADD>;
    table[index(int64, int32)] = operation<int64_t, int32_t, int64_t, Operation::ADD>;
    table[index(int64, int64)] = operation<int64_t, int64_t, int64_t, Operation::ADD>;

    table[index(int32, float32)] = operation<int32_t, float, float, Operation::ADD>;
    table[index(float32, int32)] = operation<float, int32_t, float, Operation::ADD>;
    table[index(int32, float64)] = operation<int32_t, double, double, Operation::ADD>;
    table[index(float64, int32)] = operation<double, int32_t, double, Operation::ADD>;

    table[index(float32, float32)] = operation<float, float, float, Operation::ADD>;
    table[index(float32, float64)] = operation<float, double, double, Operation::ADD>;
    table[index(float64, float32)] = operation<double, float, double, Operation::ADD>;
    table[index(float64, float64)] = operation<double, double, double, Operation::ADD>;
    return table;
}();


extern template bool operation<int8_t, int8_t, int8_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int16_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int8_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int16_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int8_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int16_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int8_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int16_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int32_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, float, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, int32_t, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, int32_t, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, float, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, float, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*); 

constexpr auto subtraction = []() {
    std::array<Kernel, index(TYPES, TYPES)> table = {[](const tensor_t*, const tensor_t*, tensor_t*) { return false; }}; 
    table[index(int8, int8)]   = operation<int8_t, int8_t, int8_t, Operation::SUB>;
    table[index(int8, int16)]  = operation<int8_t, int16_t, int16_t, Operation::SUB>; 
    table[index(int8, int32)]  = operation<int8_t, int32_t, int32_t, Operation::SUB>;
    table[index(int8, int64)]  = operation<int8_t, int64_t, int64_t, Operation::SUB>;

    table[index(int16, int8)]  = operation<int16_t, int8_t, int16_t, Operation::SUB>;
    table[index(int16, int16)] = operation<int16_t, int16_t, int16_t, Operation::SUB>;
    table[index(int16, int32)] = operation<int16_t, int32_t, int32_t, Operation::SUB>;
    table[index(int16, int64)] = operation<int16_t, int64_t, int64_t, Operation::SUB>;

    table[index(int32, int8)]  = operation<int32_t, int8_t, int32_t, Operation::SUB>;
    table[index(int32, int16)] = operation<int32_t, int16_t, int32_t, Operation::SUB>;
    table[index(int32, int32)] = operation<int32_t, int32_t, int32_t, Operation::SUB>;
    table[index(int32, int64)] = operation<int32_t, int64_t, int64_t, Operation::SUB>;

    table[index(int64, int8)]  = operation<int64_t, int8_t, int64_t, Operation::SUB>;
    table[index(int64, int16)] = operation<int64_t, int16_t, int64_t, Operation::SUB>;
    table[index(int64, int32)] = operation<int64_t, int32_t, int64_t, Operation::SUB>;
    table[index(int64, int64)] = operation<int64_t, int64_t, int64_t, Operation::SUB>;

    table[index(int32, float32)] = operation<int32_t, float, float, Operation::SUB>;
    table[index(float32, int32)] = operation<float, int32_t, float, Operation::SUB>;
    table[index(int32, float64)] = operation<int32_t, double, double, Operation::SUB>;
    table[index(float64, int32)] = operation<double, int32_t, double, Operation::SUB>;

    table[index(float32, float32)] = operation<float, float, float, Operation::SUB>;
    table[index(float32, float64)] = operation<float, double, double, Operation::SUB>;
    table[index(float64, float32)] = operation<double, float, double, Operation::SUB>;
    table[index(float64, float64)] = operation<double, double, double, Operation::SUB>;
    return table;
}(); 

extern template bool operation<int8_t, int8_t, int8_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int16_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int8_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int8_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int16_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int16_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int8_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int16_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int8_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int16_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int32_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int64_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, float, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, int32_t, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<int32_t, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, int32_t, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, float, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<float, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, float, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
extern template bool operation<double, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*); 

constexpr auto multiplication = []() {
    std::array<Kernel, index(TYPES, TYPES)> table = {[](const tensor_t*, const tensor_t*, tensor_t*) { return false; }}; 
    table[index(int8, int8)]   = operation<int8_t, int8_t, int8_t, Operation::MUL>;
    table[index(int8, int16)]  = operation<int8_t, int16_t, int16_t, Operation::MUL>; 
    table[index(int8, int32)]  = operation<int8_t, int32_t, int32_t, Operation::MUL>;
    table[index(int8, int64)]  = operation<int8_t, int64_t, int64_t, Operation::MUL>;

    table[index(int16, int8)]  = operation<int16_t, int8_t, int16_t, Operation::MUL>;
    table[index(int16, int16)] = operation<int16_t, int16_t, int16_t, Operation::MUL>;
    table[index(int16, int32)] = operation<int16_t, int32_t, int32_t, Operation::MUL>;
    table[index(int16, int64)] = operation<int16_t, int64_t, int64_t, Operation::MUL>;

    table[index(int32, int8)]  = operation<int32_t, int8_t, int32_t, Operation::MUL>;
    table[index(int32, int16)] = operation<int32_t, int16_t, int32_t, Operation::MUL>;
    table[index(int32, int32)] = operation<int32_t, int32_t, int32_t, Operation::MUL>;
    table[index(int32, int64)] = operation<int32_t, int64_t, int64_t, Operation::MUL>;

    table[index(int64, int8)]  = operation<int64_t, int8_t, int64_t, Operation::MUL>;
    table[index(int64, int16)] = operation<int64_t, int16_t, int64_t, Operation::MUL>;
    table[index(int64, int32)] = operation<int64_t, int32_t, int64_t, Operation::MUL>;
    table[index(int64, int64)] = operation<int64_t, int64_t, int64_t, Operation::MUL>;

    table[index(int32, float32)] = operation<int32_t, float, float, Operation::MUL>;
    table[index(float32, int32)] = operation<float, int32_t, float, Operation::MUL>;
    table[index(int32, float64)] = operation<int32_t, double, double, Operation::MUL>;
    table[index(float64, int32)] = operation<double, int32_t, double, Operation::MUL>;

    table[index(float32, float32)] = operation<float, float, float, Operation::MUL>;
    table[index(float32, float64)] = operation<float, double, double, Operation::MUL>;
    table[index(float64, float32)] = operation<double, float, double, Operation::MUL>;
    table[index(float64, float64)] = operation<double, double, double, Operation::MUL>;
    return table;
}();

} // namespace binary

} // namespace cpu 