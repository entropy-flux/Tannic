#include "Bindings.hpp"
#include "Tensor.hpp"
#include "Transformations.hpp" 
#include "cpu/gemm.hpp"   
#include "cpu/outer.hpp"

#ifdef CUDA
#include "cuda/gemm.cuh"   
#include "cuda/outer.cuh"
#else   
namespace cuda { 
inline status gemm(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }
inline status gemm(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }
inline status outer(tensor_t const*, tensor_t const*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
} // namespace cuda
#endif

namespace tannic::transformation { 
 
using BH = status (*)(const tensor_t*, const tensor_t*, tensor_t*);
using BD = status (*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);  
   
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

void Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    return apply<cpu::gemm, cuda::gemm>(first, second, output);
} 

void Outer::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    return apply<cpu::outer, cuda::outer>(first, second, output);
} 

} //namespace tannic::transformation