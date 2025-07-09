#pragma once

#include <array>
#include "core/tensor.h"
#include "cuda/cuda.cuh"  // Ensure this defines cuda::matmul_op

namespace {

[[noreturn]] inline void notImplemented(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t) {
    throw std::runtime_error("CUDA matmul kernel not implemented for this type combination.");
}

} // anonymous namespace

namespace cuda {

template<typename S, typename D, typename TC>
void matmul_op(const tensor_t* A, const tensor_t* B, tensor_t* C, bool A_transposed, bool B_transposed, cudaStream_t stream);

} // namespace cuda

namespace cuda::matmul {

// Full function pointer type with cudaStream_t
using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES * TYPES> table;
    table.fill(::notImplemented);

    table[index(int8, int8)]     = matmul_op<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = matmul_op<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = matmul_op<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = matmul_op<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = matmul_op<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = matmul_op<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = matmul_op<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = matmul_op<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = matmul_op<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = matmul_op<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = matmul_op<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = matmul_op<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = matmul_op<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = matmul_op<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = matmul_op<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = matmul_op<int64_t, int64_t, int64_t>;

    table[index(int32, float32)] = matmul_op<int32_t, float, float>;
    table[index(float32, int32)] = matmul_op<float, int32_t, float>;
    table[index(int32, float64)] = matmul_op<int32_t, double, double>;
    table[index(float64, int32)] = matmul_op<double, int32_t, double>;

    table[index(float32, float32)] = matmul_op<float, float, float>;
    table[index(float32, float64)] = matmul_op<float, double, double>;
    table[index(float64, float32)] = matmul_op<double, float, double>;
    table[index(float64, float64)] = matmul_op<double, double, double>;

    return table;
}();

} // namespace cuda::matmul
