#pragma once
#include <array>
#include "Types.hpp"   
 
namespace openblas::axpy {

using Kernel = void(*)(int, double, const void*, int, void*, int); 

void saxpy(
    int n,
    double alpha,
    const void* x, int incx,
    void* y, int incy
);

void daxpy(
    int n,
    double alpha,
    const void* x, int incx,
    void* y, int incy
); 


using Kernels = std::array<Kernel, TYPES>;

constexpr Kernels make_axpy_kernels() {
    Kernels kernels{};
    kernels[float32] = saxpy;
    kernels[float64] = daxpy;
    return kernels;
}

inline constexpr auto kernels = make_axpy_kernels();

} //