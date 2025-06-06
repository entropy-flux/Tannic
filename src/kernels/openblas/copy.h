#pragma once
#include <array>
#include "Types.hpp"  

namespace openblas::copy {

using Kernel = void(*)(int, const void*, int, void*, int);

void scopy(int n, const void* x, int incx, void* y, int incy);
void dcopy(int n, const void* x, int incx, void* y, int incy);

using Kernels = std::array<Kernel, TYPES>;

constexpr Kernels make_copy_kernels() {
    Kernels kernels{};
    kernels[float32] = scopy;
    kernels[float64] = dcopy;
    return kernels;
}

inline constexpr auto kernels = make_copy_kernels();

} // namespace openblas::copy
