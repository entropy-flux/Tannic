#include "Tensor.hpp"
#include "Transformations.hpp"
#include "runtime/tensor.h"
#include "cpu/cpu.hpp"  

namespace tannic { 

static inline tensor_t structure(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = reinterpret_cast<void*>(tensor.bytes()),
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
    tensor_t src1 = structure(first);
    tensor_t src2 = structure(second);
    tensor_t dst = structure(output); 
    cpu::gemm(&src1, &src2, &dst, is_transposed(first), is_transposed(second));
}

} //namespace tannic