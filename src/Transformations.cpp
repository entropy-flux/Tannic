#include "Tensor.hpp"
#include "Transformations.hpp"
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

static inline bool is_transposed(Tensor const& tensor) {
    if (tensor.rank() < 2) return false;
    Strides const& strides = tensor.strides();
    return strides[-1] > strides[-2];
}

void expression::Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize(); 
    tensor_t src1 = c_tensor_t(first);
    tensor_t src2 = c_tensor_t(second);
    tensor_t dst = c_tensor_t(output); 
    cpu::gemm(&src1, &src2, &dst, is_transposed(first), is_transposed(second));
}

void expression::Argmax::forward(Tensor const& source, Tensor& target) const {  

}

void expression::Argmin::forward(Tensor const& source, Tensor& target) const {  

}

} //namespace tannic