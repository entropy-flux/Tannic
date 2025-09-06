#include "bindings.hpp"
#include "operations.hpp"
#include "tensor.hpp" 
#include "callback.hpp"
#include "runtime/streams.h"
#include "cpu/ops.hpp"  
#ifdef CUDA
#include "cuda/ops.cuh"
#else 
namespace cuda {
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline status neg(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status add(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status mul(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status sub(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status pow(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); } 
inline status scale(const tensor_t*, const scalar_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); };     
}
#endif 
 
namespace tannic::operation {  
 
void Negation::forward(Tensor const& input, Tensor& output) const {
    Callback callback(cpu::neg, cuda::neg);
    callback(input, output);
}  

void Addition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    Callback callback(cpu::add, cuda::add);
    callback(first, second, output);
}

void Multiplication::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(cpu::mul, cuda::mul);
    callback(first, second, output);
} 

void Multiplication::forward(Tensor const& first, Scalar const& second, Tensor& output) const { 
    Callback callback(cpu::scale, cuda::scale);
    callback(first, second, output);
} 

void Subtraction::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(cpu::sub, cuda::sub);
    callback(first, second, output);
} 

void Exponentiation::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    Callback callback(cpu::pow, cuda::pow);
    callback(first, second, output);
}

} // namespace tannic