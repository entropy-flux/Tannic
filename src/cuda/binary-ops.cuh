#include <array>
#include <stdexcept>
#include "core/tensor.h"
#include "cuda/cuda.cuh"

namespace { 

[[noreturn]] inline void notImplemented(const tensor_t*, const tensor_t*, tensor_t*, auto, cudaStream_t) {
    throw std::runtime_error("CUDA Kernel not implemented for this type");
}

}


namespace cuda {

constexpr inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES)*static_cast<int>(second);
} 

template<typename S0, typename S1, typename D, typename Op>
void binaryOp(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, Op op, cudaStream_t stream = 0);

} // namespace cuda
 

namespace cuda::addition {

class Addition {
public:
    template<class A, class B>
    __device__ __host__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a + b)) -> decltype(a + b) {
        return a + b;
    }
};

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Addition, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES * TYPES> table{};
    table.fill(::notImplemented);

    // exactly mirror your CPU table entries:
    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Addition>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Addition>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Addition>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Addition>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Addition>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Addition>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Addition>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Addition>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Addition>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Addition>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Addition>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Addition>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Addition>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Addition>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Addition>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Addition>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Addition>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Addition>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Addition>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Addition>;

    table[index(float32, float32)] = binaryOp<float, float, float, Addition>;
    table[index(float32, float64)] = binaryOp<float, double, double, Addition>;
    table[index(float64, float32)] = binaryOp<double, float, double, Addition>;
    table[index(float64, float64)] = binaryOp<double, double, double, Addition>;

    return table;
}();

} // namespace addition



namespace cuda::subtraction {

class Subtraction {
public:
    template<class A, class B>
    __device__ __host__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a - b)) -> decltype(a - b) {
        return a - b;
    }
};

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Subtraction, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES * TYPES> table{};
    table.fill(::notImplemented);

    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Subtraction>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Subtraction>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Subtraction>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Subtraction>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Subtraction>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Subtraction>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Subtraction>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Subtraction>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Subtraction>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Subtraction>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Subtraction>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Subtraction>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Subtraction>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Subtraction>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Subtraction>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Subtraction>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Subtraction>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Subtraction>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Subtraction>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Subtraction>;

    table[index(float32, float32)] = binaryOp<float, float, float, Subtraction>;
    table[index(float32, float64)] = binaryOp<float, double, double, Subtraction>;
    table[index(float64, float32)] = binaryOp<double, float, double, Subtraction>;
    table[index(float64, float64)] = binaryOp<double, double, double, Subtraction>;

    return table;
}();

} // namespace cuda::subtraction


namespace cuda::multiplication {

class Multiplication {
public:
    template<class A, class B>
    __device__ __host__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a * b)) -> decltype(a * b) {
        return a * b;
    }
}; 

using Kernel = void(*)(const tensor_t*, const tensor_t*, tensor_t*, Multiplication, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES * TYPES> table{};
    table.fill(::notImplemented);

    table[index(int8, int8)]   = binaryOp<int8_t, int8_t, int8_t, Multiplication>;
    table[index(int8, int16)]  = binaryOp<int8_t, int16_t, int16_t, Multiplication>;
    table[index(int8, int32)]  = binaryOp<int8_t, int32_t, int32_t, Multiplication>;
    table[index(int8, int64)]  = binaryOp<int8_t, int64_t, int64_t, Multiplication>;

    table[index(int16, int8)]  = binaryOp<int16_t, int8_t, int16_t, Multiplication>;
    table[index(int16, int16)] = binaryOp<int16_t, int16_t, int16_t, Multiplication>;
    table[index(int16, int32)] = binaryOp<int16_t, int32_t, int32_t, Multiplication>;
    table[index(int16, int64)] = binaryOp<int16_t, int64_t, int64_t, Multiplication>;

    table[index(int32, int8)]  = binaryOp<int32_t, int8_t, int32_t, Multiplication>;
    table[index(int32, int16)] = binaryOp<int32_t, int16_t, int32_t, Multiplication>;
    table[index(int32, int32)] = binaryOp<int32_t, int32_t, int32_t, Multiplication>;
    table[index(int32, int64)] = binaryOp<int32_t, int64_t, int64_t, Multiplication>;

    table[index(int64, int8)]  = binaryOp<int64_t, int8_t, int64_t, Multiplication>;
    table[index(int64, int16)] = binaryOp<int64_t, int16_t, int64_t, Multiplication>;
    table[index(int64, int32)] = binaryOp<int64_t, int32_t, int64_t, Multiplication>;
    table[index(int64, int64)] = binaryOp<int64_t, int64_t, int64_t, Multiplication>;

    table[index(int32, float32)] = binaryOp<int32_t, float, float, Multiplication>;
    table[index(float32, int32)] = binaryOp<float, int32_t, float, Multiplication>;
    table[index(int32, float64)] = binaryOp<int32_t, double, double, Multiplication>;
    table[index(float64, int32)] = binaryOp<double, int32_t, double, Multiplication>;

    table[index(float32, float32)] = binaryOp<float, float, float, Multiplication>;
    table[index(float32, float64)] = binaryOp<float, double, double, Multiplication>;
    table[index(float64, float32)] = binaryOp<double, float, double, Multiplication>;
    table[index(float64, float64)] = binaryOp<double, double, double, Multiplication>;

    return table;
}();

} // namespace cuda::multiplication
