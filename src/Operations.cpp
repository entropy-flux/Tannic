#include "Bindings.hpp"
#include "Operations.hpp"
#include "Tensor.hpp" 
#include "cpu/cpu.hpp"  
 
namespace tannic::operation { 
 
void Negation::forward(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);  
    cpu::neg(&src, &dst);
}

void Addition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize();
    tensor_t src1 = structure(first);
    tensor_t src2 = structure(second);
    tensor_t dst = structure(output); 
    cpu::add(&src1, &src2, &dst); 
}

void Multiplication::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = structure(first);
    tensor_t src2 = structure(second);
    tensor_t dst = structure(output); 
    cpu::mul(&src1, &src2, &dst); 
}

void Subtraction::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = structure(first);
    tensor_t src2 = structure(second);
    tensor_t dst = structure(output); 
    cpu::sub(&src1, &src2, &dst); 
}

} // namespace tannic