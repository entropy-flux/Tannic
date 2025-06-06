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
// This file is part of Tannic, a C++ tensor library.

#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <variant>
#include <atomic>
#include <cstddef>
#include <vector>
#include <utility>

#include "Types.hpp"
#include "Memory/Allocators.hpp"

class Storage {
public: 

    Storage() = default;

    Storage(std::size_t size, uint8_t dsize, Allocator allocator = Host{})
        : memory_(size * dsize) 
        , allocator_(allocator) {
            references_ = new std::atomic<std::size_t>(1); 
            address_ = std::visit([&](auto& allocator) -> void* {
                return allocator.allocate(memory_);
            }, allocator_);
        }
 
    Storage(const Storage& other)
        : memory_(other.memory_) 
        , allocator_(other.allocator_)
        , address_(other.address_)
        , references_(other.references_) {
            ++(*references_);
        }
 
    Storage(Storage&& other) noexcept
        : memory_(other.memory_) 
        , allocator_(std::move(other.allocator_))
        , address_(std::exchange(other.address_, nullptr))
        , references_(std::exchange(other.references_, nullptr)) {}

 
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = other.allocator_;
            address_ = other.address_;
            references_ = other.references_;
            ++(*references_);
        }
        return *this;
    }
    
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = std::move(other.allocator_);
            address_ = std::exchange(other.address_, nullptr);
            references_ = std::exchange(other.references_, nullptr); 
        }
        return *this;
    }

    ~Storage() {
        release();
    }

    std::size_t references() const {
        return references_ ? references_->load() : 0;
    }

    void* address() { return address_; }
    void const* address() const { return address_; }
    std::size_t memory() const { return memory_; }
    Allocator const& allocator() const { return allocator_; }


private:
    void release() {
        if (references_) {
            if (--(*references_) == 0) {
                if (address_) {
                    std::visit([&](auto& variant) {
                        variant.deallocate(address_, memory_);
                    }, allocator_);
                }
                delete references_;
            }
            references_ = nullptr;
            address_ = nullptr;
        }
    }

    std::size_t memory_ = 0; 
    Allocator allocator_ = Host{};
    void* address_ = nullptr;
    std::atomic<std::size_t>* references_ = nullptr;
};
  
#endif // STORAGE_HPP