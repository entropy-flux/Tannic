#include "Bindings.hpp"
#include "Reductions.hpp"
#include "Indexing.hpp"
#include "cpu/cpu.hpp"

 
namespace tannic {
 
void expression::Argmax::forward(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);  
    cpu::argmax(&src, &dst, axis);
}

void expression::Argmin::forward(Tensor const& input, Tensor& output) const {   
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);  
    cpu::argmin(&src, &dst, axis);
}

} // namespace tannic