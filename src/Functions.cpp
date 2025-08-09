#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h"
#include "cpu/fns.hpp"

#ifdef CUDA
#include "cuda/fns.cuh"
#else
namespace cuda {
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

template <typename HFunc, typename DFunc>
static inline void apply(HFunc hcall, DFunc dcall,
                         Tensor const& input, Tensor& output) {
    allocator_t allocator = structure(input.allocator());

    switch (allocator.environment) {
        case HOST: {
            output.initialize();
            auto src = structure(input);
            auto dst = structure(output);
            auto status = hcall(&src, &dst);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break;
        }

        case DEVICE: {
            auto dvc = allocator.resource.device;
            output.initialize(Device(dvc.id));
            auto src = structure(input);
            auto dst = structure(output);
            auto stream = pop_stream(&dvc);
            auto status = dcall(&src, &dst, stream);
            put_stream(&dvc, stream);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break;
        }

        default:
            throw std::runtime_error("Unknown allocator environment");
    }
}

// ===== Operators ===== 

void Log::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::log(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::log(src, dst, stream); },
        input, output
    );
}

void Exp::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::exp(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::exp(src, dst, stream); },
        input, output
    );
}

void Sqrt::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::sqrt(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::sqrt(src, dst, stream); },
        input, output
    );
}

void Rsqrt::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [&](auto src, auto dst) { return cpu::rsqrt(src, dst, epsilon); },
        [&](auto src, auto dst, auto stream) { return cuda::rsqrt(src, dst, stream, epsilon); },
        input, output
    );
}

void Abs::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::abs(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::abs(src, dst, stream); },
        input, output
    );
}

void Sin::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::sin(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::sin(src, dst, stream); },
        input, output
    );
}

void Cos::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::cos(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::cos(src, dst, stream); },
        input, output
    );
}

void Tan::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::tan(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::tan(src, dst, stream); },
        input, output
    );
}

void Sinh::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::sinh(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::sinh(src, dst, stream); },
        input, output
    );
}

void Cosh::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::cosh(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::cosh(src, dst, stream); },
        input, output
    );
}

void Tanh::operator()(Tensor const& input, Tensor& output) const {
    apply(
        [](auto src, auto dst) { return cpu::tanh(src, dst); },
        [](auto src, auto dst, auto stream) { return cuda::tanh(src, dst, stream); },
        input, output
    );
}

} // namespace tannic::function
