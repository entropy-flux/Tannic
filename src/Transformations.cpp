#include "Bindings.hpp"
#include "Tensor.hpp"
#include "Transformations.hpp" 
#include "Callback.hpp"
#include "runtime/streams.h"
#include "cpu/gemm.hpp"   
#include "cpu/outer.hpp"
#include "cpu/reps.hpp"
#include "cpu/concat.hpp"
#include "cpu/fns.hpp"

#ifdef CUDA
#include "cuda/gemm.cuh"   
#include "cuda/outer.cuh"
#include "cuda/reps.cuh"
#include "cuda/concat.cuh"
#include "cuda/fns.cuh"
#else   
namespace cuda {  
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline status gemm(const tensor_t*, const tensor_t*, tensor_t*, stream_t, double) { throw std::runtime_error("CUDA gemm called without CUDA support"); }
inline status outer(tensor_t const*, tensor_t const*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
inline status repeat(const tensor_t*, tensor_t*, int, int, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
inline status concat(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int)  { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
inline status idn (const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
} // namespace cuda
#endif

namespace tannic::transformation {  

void Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* a, const tensor_t* b, tensor_t* out) -> status { return cpu::gemm(a, b, out, scale); },
        [&](const tensor_t* a, const tensor_t* b, tensor_t* out, stream_t stream) -> status { return cuda::gemm(a, b, out, stream, scale); }
    );
    callback(first, second, output);
} 

void Outer::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(
        [](const tensor_t* a, const tensor_t* b, tensor_t* out) -> status { return cpu::outer(a, b, out); },
        [](const tensor_t* a, const tensor_t* b, tensor_t* out, stream_t stream) -> status { return cuda::outer(a, b, out, stream); }
    );
    callback(first, second, output);
}  

void Repetition::forward(Tensor const& source, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::repeat(src, dst, axis, repeats); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::repeat(src, dst, axis, repeats, stream); }
    );
    callback(source, output);
} 

void Concatenation::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* a, const tensor_t* b, tensor_t* out) -> status { return cpu::concat(a, b, out, axis); },
        [&](const tensor_t* a, const tensor_t* b, tensor_t* out, stream_t stream) -> status { return cuda::concat(a, b, out, stream, axis); }
    );
    callback(first, second, output);
} 

void Repack::forward(Tensor const& input, Tensor& output) const {
    Callback callback(cpu::idn, cuda::idn);
    callback(input, output);
}

} // namespace tannic::transformation
