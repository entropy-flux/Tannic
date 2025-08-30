#include "bindings.hpp"
#include "functions.hpp"
#include "callback.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h"
#include "runtime/streams.h"
#include "runtime/graph.h"
#include "cpu/ops.hpp"

#ifdef CUDA
#include "cuda/ops.cuh"
#else
namespace cuda {
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline status log(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status exp(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status sqrt(const tensor_t*, tensor_t*, stream_t)               { throw std::runtime_error("CUDA not available"); }
inline status rsqrt(const tensor_t*, tensor_t*, stream_t, float)       { throw std::runtime_error("CUDA not available"); }
inline status abs(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status sin(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status cos(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status tan(const tensor_t*, tensor_t*, stream_t)                { throw std::runtime_error("CUDA not available"); }
inline status sinh(const tensor_t*, tensor_t*, stream_t)               { throw std::runtime_error("CUDA not available"); }
inline status cosh(const tensor_t*, tensor_t*, stream_t)               { throw std::runtime_error("CUDA not available"); }
inline status tanh(const tensor_t*, tensor_t*, stream_t)               { throw std::runtime_error("CUDA not available"); }
}
#endif

namespace tannic::function {
 
void Log::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::log(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::log(src, dst, stream); }
    );
    callback(input, output);
}

void Exp::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::exp(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::exp(src, dst, stream); }
    );
    callback(input, output);
}

void Sqrt::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::sqrt(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::sqrt(src, dst, stream); }
    );
    callback(input, output);
}

void Rsqrt::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::rsqrt(src, dst, epsilon); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::rsqrt(src, dst, stream, epsilon); }
    );
    callback(input, output);
}

void Abs::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::abs(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::abs(src, dst, stream); }
    );
    callback(input, output);
}

void Sin::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::sin(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::sin(src, dst, stream); }
    );
    callback(input, output);
}

void Cos::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::cos(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::cos(src, dst, stream); }
    );
    callback(input, output);
}

void Tan::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::tan(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::tan(src, dst, stream); }
    );
    callback(input, output);
}

void Sinh::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::sinh(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::sinh(src, dst, stream); }
    );
    callback(input, output);
}

void Cosh::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::cosh(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::cosh(src, dst, stream); }
    );
    callback(input, output);
}

void Tanh::operator()(Tensor const& input, Tensor& output) const {
    Callback callback(
        [](const tensor_t* src, tensor_t* dst) -> status { return cpu::tanh(src, dst); },
        [](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::tanh(src, dst, stream); }
    );
    callback(input, output);
}
 
} // namespace tannic::function
