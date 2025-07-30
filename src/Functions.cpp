#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "cpu/fns.hpp"  
#include "cuda/fns.cuh"  
  
namespace tannic::function {   

using H = void (*)(const tensor_t*, tensor_t*);
using D = void (*)(const device_t*, const tensor_t*, tensor_t*); 

template <H hcall, D dcall>
inline void apply(Tensor const& input, Tensor& output) {
    output.initialize(); 
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            hcall(&src, &dst);    
            break; 
        } 

        case DEVICE: {
            device_t dvc = allocator.resource.device;
            dcall(&dvc, &src, &dst);
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