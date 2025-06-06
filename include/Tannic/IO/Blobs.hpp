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
//
// This file is part of Tannic, a machine learning tensor library for C++.

#ifndef BLOBS_HPP
#define BLOBS_HPP

#include <variant>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string> 
#include <iomanip>
#include <cstddef> 
#include <span>  
#include <utility>

#include "Types.hpp" 
#include "Memory/Allocators.hpp"

#pragma pack(push, 1)
struct Header {
    char magic[4] = {'M', 'L', 'B', 'C'};
    uint8_t body = 0;
    uint64_t tail = 0; 

    Header() = default;

    Header(uint8_t offset, uint64_t size)
    :   body(offset)
    ,   tail(offset + size) {}

};
#pragma pack(pop)
 
template<class Object> class Metadata;

#pragma pack(push, 1)
template<>
struct Metadata<Tensor> { 
    type dtype = any;
    uint8_t rank = 0;
    uint64_t shape[8] = {};
    size_t strides[8] = {}; 

    Metadata() = default;

    template<class Shape, class Strides>
    Metadata(type dtype, const Shape& shape, const Strides& strides)
    :   dtype(dtype)
    ,   rank(static_cast<uint8_t>(shape.rank())) { 
        std::copy(shape.begin(), shape.end(), this-> shape);
        std::copy(strides.begin(), strides.end(), this-> strides);
    }
};
#pragma pack(pop)

constexpr inline uint32_t align(uint32_t offset, uint32_t alignment) {
    return (alignment == 0) ? offset : ((offset + alignment - 1) & ~(alignment - 1));
} 

class Blob { 
public:
    Blob(std::size_t size, Allocator allocator = Host{})
    :   size_(size)
    ,   allocator_(allocator) {
        bytes_ = std::visit([size = size_](auto& variant) {
            return static_cast<std::byte*>(variant.allocate(size));
        }, allocator);

        if (!bytes_) 
            throw std::bad_alloc();
    }

    Blob(std::byte* bytes, std::size_t size)
    :   bytes_(bytes)
    ,   size_(size)
    ,   allocator_(View{bytes}) {}

    ~Blob() {
        if (bytes_) {
            std::visit([this](auto& alloc) {
                alloc.deallocate(bytes_, size_);
            }, allocator_);
        }
    }
 
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;
 
    Blob(Blob&& other) noexcept
        : bytes_(std::exchange(other.bytes_, nullptr))
        , size_(std::exchange(other.size_, 0))
        , allocator_(std::move(other.allocator_)) {
    }

    Blob& operator=(Blob&& other) noexcept {
        if (this != &other) {
            if (bytes_) {
                std::visit([this](auto& alloc) {
                    alloc.deallocate(bytes_, size_);
                }, allocator_);
            }
            bytes_ = std::exchange(other.bytes_, nullptr);
            size_ = std::exchange(other.size_, 0);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }
    
    template<class Object>
    Blob(Header const& header, Object const& object, Metadata<Object> const& metadata, Allocator allocator) 
    :   size_(header.tail + sizeof(Metadata<Object>)) 
    ,   allocator_(allocator) { 
        bytes_ = std::visit([size = size_](auto& variant) {
            return static_cast<std::byte*>(variant.allocate(size));
        }, allocator);

        if (!bytes_) 
            throw std::bad_alloc();
            
        std::memcpy(bytes_, &header, header.body); 
        std::memcpy(bytes_ + header.body, object.address(), header.tail - header.body);
        std::memcpy(bytes_ + header.tail, &metadata, sizeof(Metadata<Object>));
    }
    
    std::byte* bytes() noexcept { return bytes_; }
    std::byte const* bytes() const noexcept { return bytes_; }
    std::size_t size() const noexcept { return size_; }

private: 
    std::byte* bytes_ = nullptr;
    std::size_t size_ = 0;
    Allocator allocator_;     
}; 

std::ostream& operator<<(std::ostream& os, const Blob& blob) {
    os << std::hex << std::setfill('0');
    for (size_t index = 0; index < blob.size(); ++index) {
        os << std::setw(2) << static_cast<int>(blob.bytes()[index]) << " ";
        if ((index + 1) % 16 == 0) os << "\n";
    }
    os << std::dec << "\n";
    return os;
}


#endif // BLOBS_HPP