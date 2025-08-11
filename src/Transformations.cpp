#include "Bindings.hpp"
#include "Tensor.hpp"
#include "Transformations.hpp" 
#include "runtime/streams.h"
#include "cpu/gemm.hpp"   
#include "cpu/outer.hpp"
#include "cpu/reps.hpp"
#include "cpu/concat.hpp"

#ifdef CUDA
#include "cuda/gemm.cuh"   
#include "cuda/outer.cuh"
#include "cuda/reps.cuh"
#include "cuda/concat.cuh"
#else   
namespace cuda {  
inline status gemm(const tensor_t*, const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }
inline status outer(tensor_t const*, tensor_t const*, tensor_t*, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
inline status repeat(const tensor_t*, tensor_t*, int, int, stream_t) { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
inline status concat(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int)  { throw std::runtime_error("CUDA gemm called without CUDA support"); }; 
} // namespace cuda
#endif

namespace tannic::transformation { 
 
using UH = status (*)(const tensor_t*, tensor_t*);
using UD = status (*)(const tensor_t*, tensor_t*, stream_t); 
using BH = status (*)(const tensor_t*, const tensor_t*, tensor_t*);
using BD = status (*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);   

// TODO: Refactor with variadic args and lambdas like Functions.cpp

template <UH hcall, UD dcall>
static inline void apply(Tensor const& input, Tensor& output) {   
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            output.initialize();  
            tensor_t dst = structure(output);
            tensor_t src = structure(input);
            status status = hcall(&src, &dst);    
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break; 
        } 

        case DEVICE: { 
            auto dvc = allocator.resource.device; 
            output.initialize(Device(dvc.id)); 
            tensor_t dst = structure(output);
            tensor_t src = structure(input);
            stream_t stream = pop_stream(&dvc);
            status status = dcall(&src, &dst, stream);
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

void Composition::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    return apply<cpu::gemm, cuda::gemm>(first, second, output);
} 

void Outer::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    return apply<cpu::outer, cuda::outer>(first, second, output);
} 

using RepsUH = status (*)(const tensor_t*, tensor_t*, int, int);
using RepsUD = status (*)(const tensor_t*, tensor_t*, int, int, stream_t); 

template <RepsUH hcall, RepsUD dcall>
static inline void apply(Tensor const& input, Tensor& output, int axis, int repeats) {   
    allocator_t allocator = structure(input.allocator()); 
    switch (allocator.environment) {
        case HOST: {
            output.initialize();  
            tensor_t dst = structure(output);
            tensor_t src = structure(input);
            status status = hcall(&src, &dst, axis, repeats);    
            if(status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
            break; 
        } 

        case DEVICE: { 
            auto dvc = allocator.resource.device; 
            output.initialize(Device(dvc.id)); 
            tensor_t dst = structure(output);
            tensor_t src = structure(input);
            stream_t stream = pop_stream(&dvc);
            status status = dcall(&src, &dst, axis, repeats, stream);
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

using ConcatUH = status (*)(const tensor_t*, const tensor_t*, tensor_t*, int);
using ConcatUD = status (*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int);
   
template <ConcatUH hcall, ConcatUD dcall>
static inline void applyConcat(Tensor const& first, Tensor const& second, Tensor& output, int dim) {   
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
            auto status = hcall(&src0, &src1, &dst, dim);    
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
            auto status = dcall(&src0, &src1, &dst, stream, dim);
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

void Repetition::forward(Tensor const& source, Tensor& output) const {
    apply<cpu::repeat, cuda::repeat>(source, output, indexing::normalize(axis, output.rank()), repeats); 
} 

void Concatenation::forward(Tensor const& first, Tensor const& second, Tensor& output) const {
    applyConcat<cpu::concat, cuda::concat>(first, second, output, indexing::normalize(axis, output.rank())); 
} 

} //namespace tannic::transformation