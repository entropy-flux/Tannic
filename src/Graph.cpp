#include "runtime/graph.h"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "Graph.hpp"
#include "Tensor.hpp"

namespace tannic { 

Node::Node(Tensor const& target) { 
    node_t* node = new node_t;
    node->arity = 0;
    node->priors = nullptr; 

    shape_t shape{};
    strides_t strides{};
    for (auto dim = 0; dim < target.rank(); ++dim) {
        shape.sizes[dim] = target.shape()[dim];
        strides.sizes[dim] = target.strides()[dim];
    }
 
    const Allocator& allocator = target.allocator();

    if (std::holds_alternative<Host>(allocator)) {
        Host const& resource = std::get<Host>(allocator);
        node->target = new tensor_t {
            .address = (void*)(target.bytes()),
            .rank = target.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = target.dtype(),
            .allocator = {
                .environment = HOST,
                .resource = {
                    .host = host_t {.traits = resource.pinned() ? PINNED : PAGEABLE }
                },
            }
        };
    } 
    
    else {
        Device const& resource = std::get<Device>(allocator);
        node->target = new tensor_t{
            .address = (void*)(target.bytes()),
            .rank = target.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = target.dtype(),
            .allocator = {
                .environment = DEVICE,
                .resource = {
                    .device = device_t {
                        .id = resource.id(),
                        .traits = resource.blocking() ? SYNC : ASYNC}
                },
            }
        };
    }
 
    id = reinterpret_cast<uintptr_t>(node);
}
 
Node::~Node() {
    if (id) {
        node_t* node = reinterpret_cast<node_t*>(id);
        delete node->target;     
        delete[] node->priors;  
        delete node;            
        id = 0;
    }
}

Node::Node(Node&& other) noexcept : id(other.id) {
    other.id = 0;
}

Node& Node::operator=(Node&& other) noexcept {
    if (this != &other) {
        if (id) {
            node_t* node = reinterpret_cast<node_t*>(id);
            delete node->target;
            delete[] node->priors;
            delete node;
        }
        id = other.id;
        other.id = 0;
    }
    return *this;
}
   
}