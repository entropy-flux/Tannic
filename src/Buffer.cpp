// Copyright 2025 Eric Cardozo
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
  
#include <variant>
#include <cstddef>
#include <utility>

#include "runtime/resources.h"
#include "Buffer.hpp" 

namespace tannic {

Buffer::Buffer(std::size_t nbytes, Allocator allocator)
:   nbytes_(nbytes)
,   allocator_(allocator) {
    address_ = std::visit([&](auto& variant) -> void* {
        return variant.allocate(nbytes_);
    }, allocator_);
} 

Buffer::Buffer(Buffer&& other) noexcept 
:   nbytes_(std::exchange(other.nbytes_, 0))
,   allocator_(std::move(other.allocator_))
,   address_(std::exchange(other.address_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) { 
        std::visit([&](auto& variant) {
            variant.deallocate(address_, nbytes_);
        }, allocator_);
        nbytes_ = std::exchange(other.nbytes_, 0);
        allocator_ = std::move(other.allocator_);
        address_ = std::exchange(other.address_, nullptr);
    }
    return *this;
}

Buffer::~Buffer() {  
    std::visit([&](auto& alloc) {
        alloc.deallocate(address_, nbytes_);
    }, allocator_); 
    address_ = nullptr;
    nbytes_ = 0;
}

void* Buffer::address() { 
    return address_; 
}

const void* Buffer::address() const { 
    return address_; 
}

std::size_t Buffer::nbytes() const { 
    return nbytes_; 
}

Allocator const& Buffer::allocator() const { 
    return allocator_; 
} 

environment Buffer::source() const { 
    if (std::holds_alternative<Device>(allocator_)) {
        return DEVICE;
    }  
    else {
        return HOST;
    }
} 

} // namespace TANNIC 