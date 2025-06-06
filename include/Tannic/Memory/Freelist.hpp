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
// This file is part of Tannic, a A C++ tensor library.  .

#ifndef FREELIST_HPP
#define FREELIST_HPP

#include <iostream> 
#include <type_traits>
#include <unordered_map>
#include <bit>
#include <cstdint>
#include <unordered_map>
#include <cassert> 

struct Block {
    std::size_t offset;
    std::size_t size;
    char padding[8];
    Block* free = nullptr;
};

class Freelist {
public:
    using order_type = uint8_t;
    static constexpr std::size_t header = sizeof(Block);
    static constexpr std::size_t alignment = sizeof(std::max_align_t);

    Freelist(void* buffer, std::size_t size)
    :   buffer_(buffer)
    ,   size_(size)
    ,   available_(size)
    ,   allocated_(nullptr) {}


    static inline order_type order(std::size_t size) {
        return static_cast<order_type>(std::bit_width(size - 1));
    }

    std::size_t available() const {
        return available_ >= header ? available_ - header : 0;
    }

    void* reuse(std::size_t size) {
        order_type key = order(size);
        if (reusable_[key]) {
            Block* block = reusable_[key];
            if (block->free) {
                reusable_[key] = block->free;
                block->free = nullptr;
            } else {
                reusable_.erase(key);
            }
            std::byte* address = static_cast<std::byte*>(buffer_) + block->offset;
            return address + header;
        }
        return nullptr;
    }

    void* allocate(std::size_t size) {
        if (void* reused = reuse(size)) {
            return reused;
        }

        if (size > available()) {
            return nullptr;
        }

        std::size_t offset = allocated_ ? allocated_->offset + header + allocated_->size : 0;
        std::byte* address = static_cast<std::byte*>(buffer_) + offset;
        allocated_ = new (address) Block{offset, size, {}};
        available_ -= size + header;
        return address + header;
    }

    void deallocate(void* address, std::size_t) {
        std::byte* start = static_cast<std::byte*>(address) - header;
        Block* block = reinterpret_cast<Block*>(start);
        order_type key = order(block->size);
        if (reusable_[key])
            block->free = reusable_[key];
        reusable_[key] = block;
    }

private:
    Block* allocated_;
    void* buffer_;
    std::size_t size_;
    std::size_t available_;
    std::unordered_map<order_type, Block*> reusable_;
}; 
  
#endif // FREELIST_HPP