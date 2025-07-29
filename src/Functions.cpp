#include "Bindings.hpp"
#include "Functions.hpp"
#include "runtime/tensor.h"
#include "cpu/fns.hpp"  

namespace tannic::function { 
 
void Log::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);   
    tensor_t dst = structure(output);
    cpu::log(&src, &dst);
}

void Exp::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::exp(&src, &dst);
}

void Sqrt::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sqrt(&src, &dst);
}

void Abs::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::abs(&src, &dst);
}

void Sin::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sin(&src, &dst);
}

void Cos::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::cos(&src, &dst);
}

void Tan::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::tan(&src, &dst);
}

void Sinh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::sinh(&src, &dst);
}

void Cosh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::cosh(&src, &dst);
}

void Tanh::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);
    cpu::tanh(&src, &dst);
}
 
} // namespace tannic