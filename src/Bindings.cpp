#include "Bindings.hpp"  
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"

using namespace tannic;

tensor_t structure(Tensor const& tensor) { 
    return tensor_t {
        .rank = tensor.rank(),
        .address = (void*)(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
};

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