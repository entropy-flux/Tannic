#include "Bindings.hpp"  
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"
#include "cuda/streams.cuh" 
#include "cuda/mem.cuh"

using namespace tannic;  

stream_t pop_stream(const device_t* dvc) { 
    stream_t stream;
    cuda::Streams& streams = cuda::Streams::instance();
    cudaStream_t cudaStream = streams.pop(dvc->id);
    stream.address = reinterpret_cast<uintptr_t>(cudaStream); 
    return stream;
};

void put_stream(const device_t* dvc, stream_t stream) {
    cuda::Streams& streams = cuda::Streams::instance();
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    streams.put(dvc->id, cudaStream);
} 
 
host_t structure(Host const& resource) {
    return host_t {
        .traits = resource.pinned() ? PINNED : PAGEABLE
    };
};

device_t structure(Device const& resource) {
    return device_t{
        .id = resource.id(),
        .traits = resource.blocking() ? SYNC : ASYNC  
    };
};

allocator_t structure(Allocator const& allocator) { 
    if (std::holds_alternative<Host>(allocator)) {
        Host const& resource = std::get<Host>(allocator);
        return allocator_t{
            .environment = HOST,
            .resource = {.host = structure(resource)} ,
        };
    } 

    else {
        Device const& resource = std::get<Device>(allocator);
        return allocator_t{
            .environment = DEVICE,
            .resource =  {.device = structure(resource)},
        };
    }
}; 

tensor_t structure(Tensor const& tensor) {    
    const Allocator& alloc = tensor.allocator(); 
    shape_t shape;
    strides_t strides;
    for (int dimension = 0; dimension < tensor.rank(); ++dimension) {
        shape.sizes[dimension] = tensor.shape()[dimension];
        strides.sizes[dimension] = tensor.strides()[dimension];
    }

    if (std::holds_alternative<Host>(alloc)) {
        Host const& resource = std::get<Host>(alloc);
        return tensor_t {
            .address = (void*)(tensor.bytes()),
            .rank = tensor.rank(), 
            .shape = shape,
            .strides = strides, 
            .dtype = tensor.dtype(),
            .allocator = {
                .environment = HOST,
                .resource = {.host = structure(resource)} ,
            }
        };
    } 

    else {  
        Device const& resource = std::get<Device>(alloc);
        return tensor_t {
            .address = (void*)(tensor.bytes()),
            .rank = tensor.rank(), 
            .shape = shape,
            .strides = strides, 
            .dtype = tensor.dtype(),
            .allocator = {
                .environment = DEVICE,
                .resource = {.device = structure(resource)} ,
            }
        };
    }
} 