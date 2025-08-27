#include "comparisons.hpp"
#include "callback.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h" 
#include "runtime/graph.h" 
#include "runtime/streams.h"
#include "cpu/cmps.hpp"

namespace cuda {
    
using tannic::status;
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
status eq(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status ne(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status gt(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status ge(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status lt(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status le(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}

} namespace tannic::expression { 

void EQ::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::eq, cuda::eq);
    callback(first, second, target);
}

void NE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::ne, cuda::ne);
    callback(first, second, target);
}

void GT::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::gt, cuda::gt);
    callback(first, second, target);
}

void GE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::ge, cuda::ge);
    callback(first, second, target);
}

void LT::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::lt, cuda::lt);
    callback(first, second, target);
}

void LE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::le, cuda::le);
    callback(first, second, target);
}

}