#pragma once

#include "Resources.hpp" 
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"

using namespace tannic;

inline tensor_t structure(Tensor const& tensor) { 
    return tensor_t {
        .rank = tensor.rank(),
        .address = (void*)(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
};

inline host_t structure(Host const& resource) {
    return host_t {
        .traits = resource.pinned() ? PINNED : PAGEABLE
    };
};

inline device_t structure(Device const& resource) {
    return device_t{
        .id = resource.id(),
        .traits = SYNC  
    };
};

inline allocator_t structure(Allocator const& allocator) { 
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