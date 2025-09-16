#include <stdexcept>
#include <cassert> 
#include <cstring>  
#include "resources.hpp"
#include "bindings.hpp" 
#include "runtime/streams.h"
#include "runtime/resources.h"
#ifdef CUDA
#include "cuda/mem.cuh"
#else 
namespace cuda {
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline int getDeviceCount() { throw std::runtime_error("CUDA is not available in this build"); }
inline void* allocate(const device_t*, std::size_t) { throw std::runtime_error("CUDA allocation attempted without CUDA support"); }
inline void deallocate(const device_t*, void*) { throw std::runtime_error("CUDA deallocation attempted without CUDA support"); }
} // namespace cuda
#endif


namespace tannic { 

void* Host::allocate(std::size_t nbytes) const {     
    if(pageable_) {
        return std::malloc(nbytes);
    } else {
        throw std::runtime_error("Pinned memory not supported yet!");
    }
}

void Host::deallocate(void* address, std::size_t nbytes) const {  
    if(pageable_) {
        std::free(address); 
    } else {
        throw std::runtime_error("Pinned memory not supported yet!");
    }
}

Devices::Devices() {
    count_ = cuda::getDeviceCount();
}

Device::Device(int id) 
:   id_(id) { 
    if (id < 0 || id >= Devices::count()) {
        throw std::out_of_range("Invalid device ID"); 
    } 
}

void* Device::allocate(std::size_t nbytes) const {  
    device_t resource = structure(*this); 
    return cuda::allocate(&resource, nbytes); 
}

void Device::deallocate(void* address, std::size_t nbytes) const {
    device_t resource = structure(*this);  
    cuda::deallocate(&resource, address); 
} 
 

} // namespace tannic