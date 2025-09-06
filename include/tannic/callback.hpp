// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Tensor Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef CALLBACK_HPP
#define CALLBACK_HPP

#include "runtime/tensor.h"
#include "runtime/graph.h"
#include "runtime/streams.h"
#include "runtime/resources.h"
#include "tensor.hpp"

namespace tannic {

// WARNING: THIS FILE IS UNDER ACTIVE DEVELOPMENT!.

static inline tensor_t* get_tensor(uintptr_t id) {
    return reinterpret_cast<node_t*>(id)->target;
}  

template <class H, class D>
class Callback {
    H host_fn;
    D device_fn;

public:
    Callback(H host, D device)
    :   host_fn(host)
    ,   device_fn(device) {}

    void operator()(Tensor const& input, Tensor& output) const {    
        output.initialize(input.environment());  

        if (std::holds_alternative<Host>(output.environment())) {
            tensor_t* src = get_tensor(input.id());
            tensor_t* dst = get_tensor(output.id());
            auto status = host_fn(src, dst);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            } 
        }

        else {  
            Device const& resource = std::get<Device>(output.environment());
            device_t dvc{resource.id(), resource.blocking() ? SYNC : ASYNC};
            tensor_t* src = get_tensor(input.id());
            tensor_t* dst = get_tensor(output.id());
            stream_t stream = pop_stream(&dvc);
            auto status = device_fn(src, dst, stream);
            put_stream(&dvc, stream);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            } 
        } 
    }

    void operator()(Tensor const& tensor, Scalar const& scalar, Tensor& output) {
        tensor_t* src0 = get_tensor(tensor.id());
        scalar_t  sc1{scalar.address(), scalar.dtype() };
        output.initialize(tensor.environment());  
        if (std::holds_alternative<Host>(output.environment())) {
            tensor_t* dst = get_tensor(output.id());
            auto status = host_fn(src0, &sc1, dst);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
        } 
        else {  
            Device const& resource = std::get<Device>(output.environment());
            device_t dvc{resource.id(), resource.blocking() ? SYNC : ASYNC};
            tensor_t* dst = get_tensor(output.id());
            stream_t stream = pop_stream(&dvc);
            auto status = device_fn(src0, &sc1, dst, stream);
            put_stream(&dvc, stream);
            if (status != SUCCESS) {
                throw std::runtime_error("Unsupported dtype");
            }
        }
    }


    void operator()(Tensor const& first, Tensor const& second, Tensor& output) {   
        tensor_t* src0 = get_tensor(first.id());
        tensor_t* src1 = get_tensor(second.id());
        environment_t environment;
        auto status = resolve_two_environment(&src0->environment, &src1->environment, &environment);
        if(status != SUCCESS) {
            throw std::runtime_error("Environment issue!");
        }
        switch (environment.environment) {
            case HOST: {
                host_t resource = environment.resource.host;
                output.initialize(Host());  
                tensor_t* dst = get_tensor(output.id());
                auto status = host_fn(src0, src1, dst);    
                if(status != SUCCESS) {
                    throw std::runtime_error("Unsupported dtype");
                }
                break; 
            } 

            case DEVICE: { 
                device_t dvc = environment.resource.device; 
                output.initialize(Device(dvc.id));  
                stream_t stream = pop_stream(&dvc);
                tensor_t* dst = get_tensor(output.id());
                auto status = device_fn(src0, src1, dst, stream);
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


    void operator()(Tensor const& first, Tensor const& second, Tensor const& third, Tensor& output) {   
        tensor_t* src0 = get_tensor(first.id());
        tensor_t* src1 = get_tensor(second.id());
        tensor_t* src2 = get_tensor(third.id());

        environment_t environment;
        auto status = resolve_three_environment(&src0->environment, &src1->environment,  &src2->environment, &environment);
        if(status != SUCCESS) {
            throw std::runtime_error("Environment issue !");
        }

        switch (environment.environment) {
            case HOST: {
                output.initialize(Host());  
                tensor_t* dst = get_tensor(output.id());
                auto status = host_fn(src0, src1, src2, dst);    
                if(status != SUCCESS) {
                    throw std::runtime_error("Unsupported dtype");
                }
                break; 
            } 

            case DEVICE: { 
                device_t dvc = environment.resource.device; 
                output.initialize(Device(dvc.id));  
                stream_t stream = pop_stream(&dvc);
                tensor_t* dst = get_tensor(output.id());
                auto status = device_fn(src0, src1, src2, dst, stream);
                put_stream(&dvc, stream);
                if(status != SUCCESS) {
                    throw std::runtime_error("Unsupported dtype");
                } 
                break; 
            } 
            
            default:
                throw std::runtime_error("Unsupported environment");
        }  
    } 
 
}; 

} // namespace tannic

#endif