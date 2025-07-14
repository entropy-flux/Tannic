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

#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <variant>
#include <cstddef>
#include <utility>

#include "Types.hpp"
#include "Resources.hpp"


class Storage {
public:
    Storage() = default;

    Storage(std::size_t nbytes, Allocator allocator = Host{})
    :   nbytes_(nbytes)
    ,   allocator_(allocator) {
        address_ = std::visit([&](auto& variant) -> void* {
            return variant.allocate(nbytes_);
        }, allocator_);
    }

    Storage(std::size_t size, uint8_t dsize, Allocator allocator = Host{})
    :   Storage(size * dsize, allocator) {}   
 
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    
    Storage(Storage&& other) noexcept 
    :   nbytes_(std::exchange(other.nbytes_, 0))
    ,   allocator_(std::move(other.allocator_))
    ,   address_(std::exchange(other.address_, nullptr)) {}

    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            release();
            nbytes_ = std::exchange(other.nbytes_, 0);
            allocator_ = std::move(other.allocator_);
            address_ = std::exchange(other.address_, nullptr);
        }
        return *this;
    }

    ~Storage() { 
        release();
    }
 
    void* address() { 
        return address_; 
    }

    const void* address() const { 
        return address_; 
    }

    std::size_t nbytes() const { 
        return nbytes_; 
    }

    Allocator const& allocator() const { 
        return allocator_; 
    }
     
    auto resource() const { 
        if (std::holds_alternative<Host>(allocator_)) {
            return HOST;
        } else if (std::holds_alternative<Device>(allocator_)) {
            return DEVICE;
        }
        throw std::runtime_error("Unknown allocator type");
    }

private:
    void release() {
        if (address_) {
            std::visit([&](auto& alloc) {
                alloc.deallocate(address_, nbytes_);
            }, allocator_);
            address_ = nullptr;
            nbytes_ = 0;
        }
    }

    void* address_ = nullptr;
    std::size_t nbytes_ = 0;
    Allocator allocator_ = Host{};
};

#endif // STORAGE_HPP