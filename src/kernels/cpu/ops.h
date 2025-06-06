#pragma once
#include <cstddef> 
#include <array>
#include "Types.hpp"
 
namespace cpu::negation {

void neg_float32(const void*, void*, size_t);
void neg_float64(const void*, void*, size_t); 

using Kernel = void(*)(const void*, void*, size_t);
using Kernels = std::array<Kernel, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32] = neg_float32;
    kernels[float64] = neg_float64;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::negation
 

namespace cpu::addition {

void add_float32_float32(const void*, const void*, void*, size_t);
void add_float64_float64(const void*, const void*, void*, size_t);
void add_float32_float64(const void*, const void*, void*, size_t);
void add_float64_float32(const void*, const void*, void*, size_t);

using Kernel = void(*)(const void*, const void*, void*, size_t);
using Kernels = std::array<std::array<Kernel, TYPES>, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32][float32] = add_float32_float32;
    kernels[float32][float64] = add_float32_float64;
    kernels[float64][float32] = add_float64_float32;
    kernels[float64][float64] = add_float64_float64;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::addition


namespace cpu::subtraction { 

void sub_float32_float32(const void*, const void*, void*, size_t);
void sub_float64_float64(const void*, const void*, void*, size_t);
void sub_float32_float64(const void*, const void*, void*, size_t);
void sub_float64_float32(const void*, const void*, void*, size_t);

using Kernel = void(*)(const void*, const void*, void*, size_t);
using Kernels = std::array<std::array<Kernel, TYPES>, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32][float32] = sub_float32_float32;
    kernels[float32][float64] = sub_float32_float64;
    kernels[float64][float32] = sub_float64_float32;
    kernels[float64][float64] = sub_float64_float64;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::subtraction



namespace cpu::multiplication {
  
void mul_float32_float32(const void*, const void*, void*, size_t);
void mul_float64_float64(const void*, const void*, void*, size_t);
void mul_float32_float64(const void*, const void*, void*, size_t);
void mul_float64_float32(const void*, const void*, void*, size_t);


using Kernel = void(*)(const void*, const void*, void*, size_t);
using Kernels = std::array<std::array<Kernel, TYPES>, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32][float32] = mul_float32_float32;
    kernels[float32][float64] = mul_float32_float64;
    kernels[float64][float32] = mul_float64_float32;
    kernels[float64][float64] = mul_float64_float64;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::multiplication
