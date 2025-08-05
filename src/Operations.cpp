#include "Bindings.hpp"
#include "Operations.hpp"
#include "Tensor.hpp" 
#include "cpu/ops.hpp"  
#include "cuda/ops.cuh"  
 
namespace tannic::operation {  

using UH = status (*)(const tensor_t*, tensor_t*);
using UD = status (*)(const tensor_t*, tensor_t*, stream_t); 
using BH = status (*)(const tensor_t*, const tensor_t*, tensor_t*);
using BD = status (*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t); 

template <UH hcall, UD dcall>
static inline void apply(Tensor const& input, Tensor& output) {  
    tensor_t src = structure(input);
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            output.initialize();  
            auto dst = structure(output);
            auto status = hcall(&src, &dst);    
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break; 
        } 

        case DEVICE: { 
            auto dvc = allocator.resource.device; 
            output.initialize(Device(dvc.id)); 
            auto dst = structure(output);
            auto stream = pop_stream(&dvc);
            auto status = dcall(&src, &dst, stream);
            put_stream(&dvc, stream);
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            } 
            break; 
        } 
        
        default:
            break;
        } 
}    
   
template <BH hcall, BD dcall>
static inline void apply(Tensor const& first, Tensor const& second, Tensor& output) {   
    tensor_t src0 = structure(first);
    tensor_t src1 = structure(second);
    allocator_t allocator;
    auto status = resolve(&src0.allocator, &src1.allocator, &allocator);
    if(status != SUCCESS) {
        throw std::runtime_error("Allocator issue!");
    }
    switch (allocator.environment) {
        case HOST: {
            auto resource = allocator.resource.host;
            output.initialize(Host());  
            auto dst = structure(output);
            auto status = hcall(&src0, &src1, &dst);    
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break; 
        } 

        case DEVICE: { 
            auto dvc = allocator.resource.device; 
            output.initialize(Device(dvc.id)); 
            auto dst = structure(output);
            auto stream = pop_stream(&dvc);
            auto status = dcall(&src0, &src1, &dst, stream);
            put_stream(&dvc, stream);
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            } 
            break; 
        } 
        
        default:
            break;
        } 
}     

void Negation::forward(Tensor const& input, Tensor& output) const {
    apply<cpu::neg, cuda::neg>(input, output);
}  

void Addition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    apply<cpu::add, cuda::add>(first, second, output);
}

void Multiplication::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    apply<cpu::mul, cuda::mul>(first, second, output);
}

void Subtraction::forward(Tensor const& first, Tensor const& second, Tensor& output) const { 
    apply<cpu::sub, cuda::sub>(first, second, output);
}

} // namespace tannic