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
 
#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP
 
#include <cstdint>
#include "tensor.hpp"

namespace tannic { 

constexpr uint32_t MAGIC = 0x43495245;

#pragma pack(push, 1)  
struct Header {
    uint32_t magic;      
    uint8_t  version; 
    uint16_t checksum;
    uint64_t nbytes; 
};
#pragma pack(pop) 

template<class Object> 
struct Metadata;

#pragma pack(push, 1)
template<>
struct Metadata<Tensor> { 
    uint8_t  dcode;    
    size_t   offset;  
    size_t   nbytes; 
    uint8_t  rank;    
};
#pragma pack(pop) 

inline Header headerof(Tensor const& tensor) {
    return Header{
        .magic=MAGIC,
        .version=1, 
        .checksum=0xABCD,
        .nbytes=tensor.nbytes() + sizeof(Metadata<Tensor>) + tensor.rank() * sizeof(size_t)
    };
} 

inline Metadata<Tensor> metadataof(Tensor const& tensor) {
    return Metadata<Tensor> {
        .dcode = dcodeof(tensor.dtype()),
        .offset = static_cast<size_t>(tensor.offset()),
        .nbytes = tensor.nbytes(),
        .rank = tensor.rank()
    };
} 

} // namespace TANNIC

#endif // SERIALIZATION_HPP