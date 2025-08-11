#include "Complex.hpp"
#include "Bindings.hpp"
#include "cpu/complex.hpp"
#include "runtime/streams.h"
#ifdef CUDA
#include "cuda/complex.cuh"
#else
namespace cuda {
inline status view_as_cartesian(tensor_t const*, tensor_t const*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
inline status view_as_polar(tensor_t const*, tensor_t const*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
}
#endif

namespace tannic::expression { 

using BH = status (*)(const tensor_t*, const tensor_t*, tensor_t*);
using BD = status (*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);  
   
template <BH hcall, BD dcall>
static inline void apply(Tensor const& first, Tensor const& second, Tensor& output) {   
    tensor_t src0 = structure(first);
    tensor_t src1 = structure(second);
    allocator_t allocator;
    auto status = resolve_allocator(&src0.allocator, &src1.allocator, &allocator);
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

void Cartesian::forward(Tensor const& real, Tensor const& imaginary, Tensor& complex) {
    apply<cpu::view_as_cartesian, cuda::view_as_cartesian>(real, imaginary, complex);
}

void Polar::forward(Tensor const& rho, Tensor const& theta, Tensor& complex) {
    apply<cpu::view_as_polar, cuda::view_as_polar>(rho, theta, complex);
}

}