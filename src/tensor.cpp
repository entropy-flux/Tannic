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
inline void copyFromHost(const device_t* resource, const void* src , void* dst, size_t nbytes) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); }
inline bool compareFromHost(const device_t* resource, const void* hst_ptr, const void* dvc_ptr, size_t nbytes) {throw std::runtime_error("CUDA compareFromHost called without CUDA support"); }
} // namespace cuda0
#endif

namespace tannic { 

void Tensor::initialize(Environment environment) const {   
    buffer_ = std::make_shared<Buffer>(nbytes(), environment);   
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


void Tensor::assign(bool const* value, std::ptrdiff_t index) {
    size_t byte_index = index / 8;
    uint8_t bit_index = index % 8;

    std::byte* target = static_cast<std::byte*>(buffer_->address()) + byte_index;

    if (std::holds_alternative<Host>(this->environment())) {
        if (*value) {
            *target |= std::byte(1u << bit_index);  
        } else {
            *target &= std::byte(~(1u << bit_index)); 
        }
    }
    else {
        throw std::runtime_error("Unimplemented");
    }
} 

bool Tensor::compare(std::byte const* hst_ptr, std::ptrdiff_t offset) const {   
    void const* lcl_ptr = static_cast<std::byte const*>(buffer_->address()) + offset; 
    environment_t environment = structure(this->environment()); 
    if (std::holds_alternative<Host>(this->environment())) {  
        return std::memcmp(hst_ptr, lcl_ptr, dsizeof(dtype_)) == 0;  
    } 
    else {
        device_t device = structure(std::get<Device>(this->environment()));
        return cuda::compareFromHost(&device, (void const*)(hst_ptr), lcl_ptr, dsizeof(dtype_));
    }
}  

} // namespace tannic