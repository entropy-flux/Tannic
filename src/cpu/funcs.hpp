#pragma once
#include <array>
#include <cmath>  
#include <stdexcept>
#include "ctypes/types.h"
#include "ctypes/tensor.h" 

namespace cpu {

enum class Function {
    LOG,
    EXP,
    TAN,
    SQRT,
    ABS,
    SIN,
    COS,
    SINH,
    COSH,
    TANH
};

static constexpr inline int index(type t) {
    return static_cast<int>(t);
}

template<typename S, typename T, Function F>
bool function(const tensor_t*, tensor_t*);  

using Kernel = bool(*)(const tensor_t*, tensor_t*);    

extern template bool function<float, float, Function::LOG>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::LOG>(const tensor_t*, tensor_t*); 
 
constexpr auto log = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::LOG>;
    table[index(float64)] = function<double, double, Function::LOG>;
    return table;
}();


extern template bool function<float, float, Function::EXP>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::EXP>(const tensor_t*, tensor_t*); 

constexpr auto exp = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::EXP>;
    table[index(float64)] = function<double, double, Function::EXP>;
    return table;
}();


extern template bool function<float, float, Function::TAN>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::TAN>(const tensor_t*, tensor_t*); 

constexpr auto tan = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::TAN>;
    table[index(float64)] = function<double, double, Function::TAN>;
    return table;
}();

extern template bool function<float, float, Function::SQRT>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::SQRT>(const tensor_t*, tensor_t*); 

constexpr auto sqrt = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::SQRT>;
    table[index(float64)] = function<double, double, Function::SQRT>;
    return table;
}();

extern template bool function<float, float, Function::ABS>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::ABS>(const tensor_t*, tensor_t*);  

constexpr auto abs = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::ABS>;
    table[index(float64)] = function<double, double, Function::ABS>;
    return table;
}();

 
extern template bool function<float, float, Function::SIN>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::SIN>(const tensor_t*, tensor_t*);  

constexpr auto sin = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::SIN>;
    table[index(float64)] = function<double, double, Function::SIN>;
    return table;
}();

extern template bool function<float, float, Function::COS>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::COS>(const tensor_t*, tensor_t*); 

constexpr auto cos = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::COS>;
    table[index(float64)] = function<double, double, Function::COS>;
    return table;
}();


extern template bool function<float, float, Function::SINH>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::SINH>(const tensor_t*, tensor_t*); 

constexpr auto sinh = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::SINH>;
    table[index(float64)] = function<double, double, Function::SINH>;
    return table;
}();


extern template bool function<float, float, Function::COSH>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::COSH>(const tensor_t*, tensor_t*); 

constexpr auto cosh = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::COSH>;
    table[index(float64)] = function<double, double, Function::COSH>;
    return table;
}();


extern template bool function<float, float, Function::TANH>(const tensor_t*, tensor_t*);
extern template bool function<double, double, Function::TANH>(const tensor_t*, tensor_t*); 

constexpr auto tanh = []() {
    std::array<Kernel, index(TYPES)> table = {[](const tensor_t*, tensor_t*) { return false; }};
    table[index(float32)] = function<float, float, Function::TANH>;
    table[index(float64)] = function<double, double, Function::TANH>;
    return table;
}(); 
 
} // namespace cpu