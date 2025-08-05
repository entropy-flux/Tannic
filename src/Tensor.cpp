#include "Bindings.hpp" 
#include "Types.hpp"
#include "Tensor.hpp"   
#include <ostream>
#include <cstring>
#include <functional>

#ifdef CUDA
#include "cuda/mem.cuh"
#else 
namespace cuda {
inline void copyFromHost(const device_t*, std::byte const* src, std::byte* dst, size_t size) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); }
inline bool compareFromHost(const device_t*, std::byte const* src, std::byte const* dst, size_t size) { throw std::runtime_error("CUDA compareFromHost called without CUDA support"); }
} // namespace cuda
#endif

namespace tannic {

void Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {      
    std::byte* target = static_cast<std::byte*>(buffer_->address()) + offset;    
    if (std::holds_alternative<Host>(this->allocator())) {
        std::memcpy(target, value, dsizeof(dtype_)); 
    } 
    
    else {
        device_t device = structure(std::get<Device>(this->allocator()));
        cuda::copyFromHost(&device, value, target, dsizeof(dtype_));
    }
} 

bool Tensor::compare(std::byte const* hst_ptr, std::ptrdiff_t offset) const {  
    std::byte const* dvc_ptr = static_cast<std::byte const*>(buffer_->address()) + offset;  
    allocator_t allocator = structure(this->allocator()); 
    if (std::holds_alternative<Host>(this->allocator())) {
        return std::memcmp(hst_ptr, dvc_ptr, dsizeof(dtype_)) == 0;  
    } 

    else {
        device_t device = structure(std::get<Device>(this->allocator()));
        return cuda::compareFromHost(&device, hst_ptr, dvc_ptr, dsizeof(dtype_));
    }
}  

} // namespace tannic