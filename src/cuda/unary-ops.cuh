#pragma once

#include <array>
#include <cmath>
#include <cuda_runtime.h>

#include "core/tensor.h"
#include "cuda/cuda.cuh"  
 
static inline void notImplemented(const tensor_t*, tensor_t*, auto, cudaStream_t) {
    throw std::runtime_error("Kernel not implemented for this type");
}

static constexpr inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES)*static_cast<int>(second);
} 

namespace cuda {

// General unaryOp template
template<typename S, typename D, typename Op>
void unaryOp(const tensor_t* A, tensor_t* B, Op op, cudaStream_t stream);

} // namespace cuda


namespace cuda::negation_op {

class Negation {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(-a)) {
        return -a;
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Negation, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[int8]    = cuda::unaryOp<int8_t, int8_t, Negation>;
    table[int16]   = cuda::unaryOp<int16_t, int16_t, Negation>;
    table[int32]   = cuda::unaryOp<int32_t, int32_t, Negation>;
    table[int64]   = cuda::unaryOp<int64_t, int64_t, Negation>;
    table[float32] = cuda::unaryOp<float, float, Negation>;
    table[float64] = cuda::unaryOp<double, double, Negation>;
    return table;
}();

} // namespace cuda::negation

// --------------------------- Log ---------------------------

namespace cuda::log_op {

class Log {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::log(a))) {
        return ::log(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Log, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Log>;
    table[float64] = cuda::unaryOp<double, double, Log>;
    return table;
}();

} // namespace cuda::log_op

// --------------------------- Exp ---------------------------

namespace cuda::exp_op {

class Exp {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::exp(a))) {
        return ::exp(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Exp, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Exp>;
    table[float64] = cuda::unaryOp<double, double, Exp>;
    return table;
}();

} // namespace cuda::exp_op

// --------------------------- Sqrt ---------------------------

namespace cuda::sqrt_op {

class Sqrt {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::sqrt(a))) {
        return ::sqrt(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sqrt, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Sqrt>;
    table[float64] = cuda::unaryOp<double, double, Sqrt>;
    return table;
}();

} // namespace cuda::sqrt_op

// --------------------------- Abs ---------------------------

namespace cuda::abs_op {

class Abs {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::abs(a))) {
        return ::abs(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Abs, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[int8]    = cuda::unaryOp<int8_t, int8_t, Abs>;
    table[int16]   = cuda::unaryOp<int16_t, int16_t, Abs>;
    table[int32]   = cuda::unaryOp<int32_t, int32_t, Abs>;
    table[int64]   = cuda::unaryOp<int64_t, int64_t, Abs>;
    table[float32] = cuda::unaryOp<float, float, Abs>;
    table[float64] = cuda::unaryOp<double, double, Abs>;
    return table;
}();

} // namespace cuda::abs_op

// --------------------------- Sin ---------------------------

namespace cuda::sin_op {

class Sin {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::sin(a))) {
        return ::sin(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sin, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Sin>;
    table[float64] = cuda::unaryOp<double, double, Sin>;
    return table;
}();

} // namespace cuda::sin_op

// --------------------------- Cos ---------------------------

namespace cuda::cos_op {

class Cos {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::cos(a))) {
        return ::cos(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Cos, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Cos>;
    table[float64] = cuda::unaryOp<double, double, Cos>;
    return table;
}();

} // namespace cuda::cos_op

// --------------------------- Tan ---------------------------

namespace cuda::tan_op {

class Tan {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::tan(a))) {
        return ::tan(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Tan, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Tan>;
    table[float64] = cuda::unaryOp<double, double, Tan>;
    return table;
}();

} // namespace cuda::tan_op

// --------------------------- Sinh ---------------------------

namespace cuda::sinh_op {

class Sinh {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::sinh(a))) {
        return ::sinh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sinh, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Sinh>;
    table[float64] = cuda::unaryOp<double, double, Sinh>;
    return table;
}();

} // namespace cuda::sinh_op

// --------------------------- Cosh ---------------------------

namespace cuda::cosh_op {

class Cosh {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::cosh(a))) {
        return ::cosh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Cosh, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Cosh>;
    table[float64] = cuda::unaryOp<double, double, Cosh>;
    return table;
}();

} // namespace cuda::cosh_op

// --------------------------- Tanh ---------------------------

namespace cuda::tanh_op {

class Tanh {
public:
    template<class A>
    __device__ auto operator()(A a) const noexcept(noexcept(::tanh(a))) {
        return ::tanh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Tanh, cudaStream_t);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(notImplemented);
    table[float32] = cuda::unaryOp<float, float, Tanh>;
    table[float64] = cuda::unaryOp<double, double, Tanh>;
    return table;
}();

} // namespace cuda::tanh_op