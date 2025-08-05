#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h"
#include "cpu/fns.hpp"  
#include "cuda/fns.cuh"  
  
namespace tannic::function {   

using H = status (*)(const tensor_t*, tensor_t*);
using D = status (*)(const tensor_t*, tensor_t*, stream_t); 

struct layout_t {
    uint8_t rank;
    uint32_t shape;
    int64_t strides;
};

template <H hcall, D dcall>
inline void apply(Tensor const& input, Tensor& output) {  
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