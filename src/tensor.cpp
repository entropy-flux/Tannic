#include "bindings.hpp" 
#include "types.hpp"
#include "tensor.hpp"
#include "runtime/streams.h"   
#include <ostream>
#include <cstring>
#include <functional> 
#ifdef CUDA
#include "cuda/mem.cuh" 
#else 
namespace cuda {
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline void copyFromHost(const device_t*, std::byte const* src, std::byte* dst, size_t size) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); }
inline bool compareFromHost(const device_t*, std::byte const* src, std::byte const* dst, size_t size) { throw std::runtime_error("CUDA compareFromHost called without CUDA support"); }
} // namespace cuda
#endif

namespace tannic { 

void Tensor::initialize(Environment environment) const {  
    buffer_ = std::make_shared<Buffer>(nbytes_, environment);  
    node_ = std::make_shared<Node>(*this);
}    

void Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {      
    std::byte* target = static_cast<std::byte*>(buffer_->address()) + offset;    
    if (std::holds_alternative<Host>(this->environment())) {
        std::memcpy(target, value, dsizeof(dtype_)); 
    } 
    
    else {
        device_t device = structure(std::get<Device>(this->environment()));
        cuda::copyFromHost(&device, value, target, dsizeof(dtype_));
    }
} 

bool Tensor::compare(std::byte const* hst_ptr, std::ptrdiff_t offset) const {  
    std::byte const* dvc_ptr = static_cast<std::byte const*>(buffer_->address()) + offset;  
    environment_t environment = structure(this->environment()); 
    if (std::holds_alternative<Host>(this->environment())) {
        return std::memcmp(hst_ptr, dvc_ptr, dsizeof(dtype_)) == 0;  
    } 

    else {
        device_t device = structure(std::get<Device>(this->environment()));
        return cuda::compareFromHost(&device, hst_ptr, dvc_ptr, dsizeof(dtype_));
    }
}  

} // namespace tannic