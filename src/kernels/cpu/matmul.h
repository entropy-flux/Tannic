#pragma once
#include <cstddef> 
#include <array>
#include "Types.hpp"

#pragma once
#include <cstddef>
#include <array>
#include "Types.hpp"
 

namespace cpu::linear {

using Kernel = void(*)(
    bool, bool, bool, // is_rowMajor, A_is_transposed B_is_transposed
    int, int, int,    // M, N, K, 
    const void*, int, // A, lda     
    const void*, int, // B, ldb
    void*, int        // C, ldc
);

void linear_float32_float32(
    bool, bool, bool,
    int, int, int, 
    const void*, int,
    const void*, int, 
    void*, int
);

void linear_float32_float64(
    bool, bool, bool,
    int, int, int, 
    const void*, int,
    const void*, int, 
    void*, int
);

void linear_float64_float32(
    bool, bool, bool,
    int, int, int, 
    const void*, int,
    const void*, int, 
    void*, int
);

void linear_float64_float64(
    bool, bool, bool,
    int, int, int, 
    const void*, int,
    const void*, int, 
    void*, int
);

using Kernels = std::array<std::array<Kernel, TYPES>, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32][float32] = linear_float32_float32;
    kernels[float32][float64] = linear_float32_float64;
    kernels[float64][float32] = linear_float64_float32;
    kernels[float64][float64] = linear_float64_float64;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::linear
