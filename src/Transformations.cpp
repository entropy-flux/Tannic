#include "Tensor.hpp"
#include "Transformations.hpp"
#include "ctypes/tensor.h"
#include "cpu/gemm.hpp"

static inline tensor_t ctensor(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = reinterpret_cast<void*>(tensor.buffer()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
}

static inline bool is_transposed(Tensor const& tensor) {
    if (tensor.rank() < 2) return false;
    Strides const& strides = tensor.strides();
    return strides[-1] > strides[-2];
}

void expression::Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize();
    tensor_t src1 = ctensor(first);
    tensor_t src2 = ctensor(second);
    tensor_t dst = ctensor(output); 
    bool success = cpu::matmul[cpu::index(first.dtype(), second.dtype())](&src1, &src2, &dst, is_transposed(first), is_transposed(second)); 
    if (!success) {
        throw std::runtime_error(
            "Matrix multiplication failed for dtypes: " +
            dnameof(first.dtype()) + " and " +
            dnameof(second.dtype())
        );
    }
}