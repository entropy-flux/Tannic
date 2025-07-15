#include "Tensor.hpp"
#include "Transformations.hpp"
#include "core/tensor.h"
#include "cpu/gemm.hpp" 

using namespace tannic;

static inline tensor_t c_tensor_t(Tensor const& tensor, bool is_transposed = false) {
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
    bool src1_transposed = is_transposed(first);
    bool src2_transposed = is_transposed(second);
    tensor_t src1 = c_tensor_t(first, src1_transposed);
    tensor_t src2 = c_tensor_t(second, src1_transposed);
    tensor_t dst = c_tensor_t(output); 
    bool success = cpu::matmul[cpu::index(first.dtype(), second.dtype())](&src1, &src2, &dst, is_transposed(first), is_transposed(second)); 
    if (!success) {
        throw std::runtime_error(
            "Matrix multiplication failed for dtypes: " +
            dnameof(first.dtype()) + " and " +
            dnameof(second.dtype())
        );
    }
}