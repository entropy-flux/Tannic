#include "Bindings.hpp"
#include "Tensor.hpp"
#include "Transformations.hpp" 
#include "cpu/gemm.hpp"   

namespace tannic {  

void expression::Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    output.initialize(); 
    tensor_t src1 = structure(first);
    tensor_t src2 = structure(second);
    tensor_t dst = structure(output); 
    cpu::gemm(&src1, &src2, &dst);
}

} //namespace tannic