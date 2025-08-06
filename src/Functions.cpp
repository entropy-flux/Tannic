#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h"
#include "cpu/fns.hpp"   
#ifdef CUDA
#include "cuda/fns.cuh"
#else 
namespace cuda {
inline status log(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status exp(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status sqrt(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status abs(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status sin(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status cos(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status tan(const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status sinh(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status cosh(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status tanh(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
}
#endif
  
namespace tannic::function {   

using H = status (*)(const tensor_t*, tensor_t*);
using D = status (*)(const tensor_t*, tensor_t*, stream_t);   

template <H hcall, D dcall>
static inline void apply(Tensor const& input, Tensor& output) {  
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            output.initialize(); 
            auto src = structure(input);
            auto dst = structure(output);
            auto status = hcall(&src, &dst);    
            if(status != SUCCESS) {
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
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            } 
            break; 
        } 
        
        default:
            break;
        } 
}  
 
void Log::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::log, cuda::log>(input, output);
}

void Exp::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::exp, cuda::exp>(input, output);
}

void Sqrt::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::sqrt, cuda::sqrt>(input, output);
}

void Abs::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::abs, cuda::abs>(input, output);
}

void Sin::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::sin, cuda::sin>(input, output);
}

void Cos::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::cos, cuda::cos>(input, output);
}

void Tan::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::tan, cuda::tan>(input, output);
}

void Sinh::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::sinh, cuda::sinh>(input, output);
}

void Cosh::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::cosh, cuda::cosh>(input, output);
}

void Tanh::operator()(Tensor const& input, Tensor& output) const {
    apply<cpu::tanh, cuda::tanh>(input, output);
}
 
} // namespace tannic