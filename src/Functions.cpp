#include "Functions.hpp"
#include "runtime/tensor.h"
#include "cpu/cpu.hpp"  

namespace tannic { 

static inline tensor_t c_tensor_t(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = static_cast<void*>(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
}
 
void expression::Log::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    
    tensor_t dst = c_tensor_t(output);
    cpu::log(&src, &dst);
}

void expression::Exp::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::exp(&src, &dst);
}

void expression::Sqrt::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::sqrt(&src, &dst);
}

void expression::Abs::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::abs(&src, &dst);
}

void expression::Sin::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::sin(&src, &dst);
}

void expression::Cos::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::cos(&src, &dst);
}

void expression::Tan::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::tan(&src, &dst);
}

void expression::Sinh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::sinh(&src, &dst);
}

void expression::Cosh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::cosh(&src, &dst);
}

void expression::Tanh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::tanh(&src, &dst);
}
 
} // namespace tannic