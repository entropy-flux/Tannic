// Copyright 2025 Eric Cardozo
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

#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "Tensor.hpp"
#include "Blobs.hpp"

/*
TODO: Serialization for Parameters and Modules. 
*/

Blob serialize(Tensor const& tensor, uint32_t alignment = 0, Allocator allocator = Host{}) {
    Header header(align(sizeof(Header), alignment), tensor.storage().memory());  
    Metadata<Tensor> metadata(tensor.dtype(), tensor.shape(), tensor.strides());
    return Blob(header, tensor, metadata, allocator);
} 

Tensor deserialize(Blob const& blob) {
    Header header; 
    std::memcpy(&header, blob.bytes(), sizeof(Header));
    assert(std::string(header.magic, 4) == "MLBC" && "Invalid magic number");

    Metadata<Tensor> metadata;  
    std::memcpy(&metadata, blob.bytes() + header.tail, sizeof(Metadata<Tensor>));
    void* address = const_cast<void*>(static_cast<void const*>(blob.bytes() + header.body));
    Storage storage(metadata.rank, dsizeof(metadata.dtype), View{address});  

    std::span<Shape::size_type> shape(metadata.shape, metadata.rank);
    std::span<Strides::step_type> strides(metadata.strides, metadata.rank); 
    return Tensor(storage, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), metadata.dtype, 0);
} 

#endif // SERIALIZATION_HPP
