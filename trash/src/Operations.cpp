#include "Operations.hpp"
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "cpu/ops.hpp"  
 
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
    bool status = cpu::unary::negation[cpu::unary::index(input.dtype())](&src, &dst);
    if (!status) {
        throw std::runtime_error(
            "Negation operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void operation::Addition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output);
    bool success = cpu::binary::addition[cpu::binary::index(first.dtype(), second.dtype())](&src1, &src2, &dst);
    if (!success) {
        throw std::runtime_error(
            "Addition operation failed for dtypes: " +
            dnameof(first.dtype()) + " and " +
            dnameof(second.dtype())
        );
    }
}

void operation::Multiplication::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output);
    bool status = cpu::binary::multiplication[cpu::binary::index(first.dtype(), second.dtype())](&src1, &src2, &dst);
    if (!status) {
        throw std::runtime_error(
            "Multiplication operation failed for dtypes: " +
            dnameof(first.dtype()) + " and " +
            dnameof(second.dtype())
        );
    }
}

void operation::Subtraction::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    output.initialize();
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output); 
    bool status = cpu::binary::subtraction[cpu::binary::index(first.dtype(), second.dtype())](&src1, &src2, &dst);
    if (!status) {
        throw std::runtime_error(
            "Subtraction operation failed for dtypes: " +
            dnameof(first.dtype()) + " and " +
            dnameof(second.dtype())
        );
    }
}

} // namespace tannic