#include "Bindings.hpp"
#include "Reductions.hpp"
#include "Indexing.hpp"
#include "cpu/argcmp.hpp" 
 
namespace tannic {
 
void expression::Argmax::forward(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);  
    auto status = cpu::argmax(&src, &dst, axis);
    if(status != SUCCESS) {
        throw std::runtime_error("Unsupported dtype");
    } 
}

void expression::Argmin::forward(Tensor const& input, Tensor& output) const {   
    output.initialize();
    tensor_t src = structure(input);
    tensor_t dst = structure(output);  
    auto status = cpu::argmin(&src, &dst, axis);
    if(status != SUCCESS) {
        throw std::runtime_error("Unsupported dtype");
    } 
}

} // namespace tannic