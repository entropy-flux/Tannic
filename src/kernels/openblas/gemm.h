#pragma once
#include <array>
#include "Types.hpp"   

namespace openblas {
    
enum class Order {
    RowMajor = 101,
    ColMajor = 102, 
};

enum class Transposed {
    False = 111,
    True = 112,
};

}

namespace openblas::gemm {

using Kernel = void(*)(
    Order, Transposed, Transposed,
    int, int, int,
    double,
    const void*, int,
    const void*, int,
    double,
    void*, int
);

void sgemm(
    Order Order, Transposed TransA, Transposed TransB,
    int M, int N, int K,
    double alpha, 
    const void* A, int lda,
    const void* B, int ldb,
    double beta,
    void* C, int ldc
);

void dgemm(
    Order Order, Transposed TransA, Transposed TransB,
    int M, int N, int K,
    double alpha, 
    const void* A, int lda,
    const void* B, int ldb,
    double beta,
    void* C, int ldc
);

using Kernels = std::array<std::array<Kernel, TYPES>, TYPES>;

constexpr Kernels make_kernels() {
    Kernels kernels{};
    kernels[float32][float32] = sgemm; 
    kernels[float64][float64] = dgemm;
    return kernels;
}

inline constexpr auto kernels = make_kernels();

} // namespace cpu::gemm