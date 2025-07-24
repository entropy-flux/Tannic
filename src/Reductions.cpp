#include "Reductions.hpp"
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

void expression::Argmax::forward(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);  
    cpu::argmax(&src, &dst);
}

void expression::Argmin::forward(Tensor const& input, Tensor& output) const {   
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);  
    cpu::argmin(&src, &dst);
}

} // namespace tannic