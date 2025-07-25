#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "cpu/cpu.hpp"  

namespace tannic { 
 
void expression::Log::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);   
    tensor_t dst = structure(output);
    cpu::log(&src, &dst);
}

void expression::Exp::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::exp(&src, &dst);
}

void expression::Sqrt::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sqrt(&src, &dst);
}

void expression::Abs::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::abs(&src, &dst);
}

void expression::Sin::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sin(&src, &dst);
}

void expression::Cos::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::cos(&src, &dst);
}

void expression::Tan::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::tan(&src, &dst);
}

void expression::Sinh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sinh(&src, &dst);
}

void expression::Cosh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::cosh(&src, &dst);
}

void expression::Tanh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::tanh(&src, &dst);
}
 
} // namespace tannic