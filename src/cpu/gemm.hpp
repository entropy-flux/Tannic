#pragma once

#include <array> 
#include "ctypes/tensor.h" 
      
namespace cpu {

static constexpr inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES)*static_cast<int>(second);
} 

template<typename S1, typename S2, typename D>
bool gemm(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
 
extern template bool gemm<int8_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int8_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int8_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int8_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

extern template bool gemm<int16_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int16_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int16_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int16_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

extern template bool gemm<int32_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int32_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int32_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int32_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

extern template bool gemm<int64_t, int8_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int64_t, int16_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int64_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int64_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

extern template bool gemm<int32_t, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<float, int32_t, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<int32_t, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<double, int32_t, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

extern template bool gemm<float, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<float, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<double, float, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
extern template bool gemm<double, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

using Kernel = bool(*)(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

constexpr auto matmul = []() {
    std::array<Kernel, index(TYPES,TYPES)> table {[](const tensor_t*, const tensor_t*, tensor_t*, bool, bool) { return false; }};  
    table[index(int8, int8)]     = gemm<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = gemm<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = gemm<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = gemm<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = gemm<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = gemm<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = gemm<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = gemm<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = gemm<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = gemm<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = gemm<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = gemm<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = gemm<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = gemm<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = gemm<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = gemm<int64_t, int64_t, int64_t>;

    table[index(int32, float32)] = gemm<int32_t, float, float>;
    table[index(float32, int32)] = gemm<float, int32_t, float>;
    table[index(int32, float64)] = gemm<int32_t, double, double>;
    table[index(float64, int32)] = gemm<double, int32_t, double>;

    table[index(float32, float32)] = gemm<float, float, float>;
    table[index(float32, float64)] = gemm<float, double, double>;
    table[index(float64, float32)] = gemm<double, float, double>;
    table[index(float64, float64)] = gemm<double, double, double>;
    return table;
}();

} // namespace cpu