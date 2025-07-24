#include "Operations.hpp"
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "cpu/cpu.hpp"  
 
namespace tannic { 

static inline tensor_t c_tensor_t(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = reinterpret_cast<void*>(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
}

void operation::Negation::forward(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);  
    cpu::neg(&src, &dst);
}

void operation::Addition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output); 
    cpu::add(&src1, &src2, &dst); 
}

void operation::Multiplication::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output); 
    cpu::mul(&src1, &src2, &dst); 
}

void operation::Subtraction::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output); 
    cpu::sub(&src1, &src2, &dst); 
}

} // namespace tannic