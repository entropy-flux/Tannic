#pragma once
#include <array>
#include "Types.hpp"  

namespace openblas::scal {

using Kernel = void(*)(int, double, void*, int);

void sscal(
    int n,
    double alpha,
    void* x, int incx
);

void dscal(
    int n,
    double alpha,
    void* x, int incx
);

using Kernels = std::array<Kernel, TYPES>;

constexpr Kernels make_scal_kernels() {
    Kernels kernels{};
    kernels[float32] = sscal;
    kernels[float64] = dscal;
    return kernels;
}

inline constexpr auto kernels = make_scal_kernels();

} // namespace openblas::scal
