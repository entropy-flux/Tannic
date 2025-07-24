#pragma once
#include "runtime/types.h"
#include "runtime/tensor.h"
#include <array>

namespace cpu {

enum class Argcmp {
    ARGMAX,
    ARGMIN,
}; 

static constexpr inline int index(type type) {
    return static_cast<int>(type);
}
 
using Compare = bool(*)(const tensor_t*, tensor_t*, int64_t, bool);    

template<typename S, Argcmp A>
bool argcmp(const tensor_t*, tensor_t*, int64_t dim = -1, bool keepdim = false); 

// Updated extern templates with additional parameters
extern template bool argcmp<int8_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int16_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int32_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int64_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<float, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<double, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);

constexpr auto argmax = []() {  
    std::array<Compare, index(TYPES)> table = {[](const tensor_t*, tensor_t*, int64_t, bool) { return false; }};
    table[index(int8)] = argcmp<int8_t, Argcmp::ARGMAX>;
    table[index(int16)] = argcmp<int16_t, Argcmp::ARGMAX>;
    table[index(int32)] = argcmp<int32_t, Argcmp::ARGMAX>;
    table[index(int64)] = argcmp<int64_t, Argcmp::ARGMAX>;
    table[index(float32)] = argcmp<float, Argcmp::ARGMAX>;
    table[index(float64)] = argcmp<double, Argcmp::ARGMAX>;
    return table;
}();

extern template bool argcmp<int8_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int16_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int32_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<int64_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<float, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
extern template bool argcmp<double, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);

constexpr auto argmin = []() {  
    std::array<Compare, index(TYPES)> table = {[](const tensor_t*, tensor_t*, int64_t, bool) { return false; }};
    table[index(int8)] = argcmp<int8_t, Argcmp::ARGMIN>;
    table[index(int16)] = argcmp<int16_t, Argcmp::ARGMIN>;
    table[index(int32)] = argcmp<int32_t, Argcmp::ARGMIN>;
    table[index(int64)] = argcmp<int64_t, Argcmp::ARGMIN>;
    table[index(float32)] = argcmp<float, Argcmp::ARGMIN>;
    table[index(float64)] = argcmp<double, Argcmp::ARGMIN>;
    return table;
}();  

} // namespace cpu
 