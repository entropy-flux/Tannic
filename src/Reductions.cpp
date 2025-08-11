#include "Bindings.hpp"
#include "Reductions.hpp"
#include "Indexing.hpp"
#include "runtime/streams.h"
#include "cpu/argcmp.hpp" 
#ifdef CUDA
#include "cuda/argcmp.cuh"
#else 
namespace cuda { 
inline status argmax(const tensor_t*, tensor_t*, uint8_t, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline status argmin(const tensor_t*, tensor_t*, uint8_t, stream_t)  { throw std::runtime_error("CUDA not available"); } 
}
#endif 

namespace tannic {   
  
using H = status (*)(const tensor_t*, tensor_t*, uint8_t);
using D = status (*)(const tensor_t*, tensor_t*, uint8_t, stream_t);   

template <H hcall, D dcall>
static inline void apply(Tensor const& input, Tensor& output, uint8_t axis) {  
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            output.initialize(); 
            auto src = structure(input);
            auto dst = structure(output);
            auto status = hcall(&src, &dst, axis);    
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break; 
        } 

        case DEVICE: { 
            auto dvc = allocator.resource.device; 
            output.initialize(Device(dvc.id));
            auto src = structure(input);
            auto dst = structure(output);
            auto stream = pop_stream(&dvc);
            auto status = dcall(&src, &dst, axis, stream);
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
 
void expression::Argmax::forward(Tensor const& input, Tensor& output) const {  
    apply<cpu::argmax, cuda::argmax>(input, output, axis);
}

void expression::Argmin::forward(Tensor const& input, Tensor& output) const {   
    apply<cpu::argmin, cuda::argmin>(input, output, axis);
}

} // namespace tannic