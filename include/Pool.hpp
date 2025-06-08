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

#ifndef POOL_HPP
#define POOL_HPP
#include <type_traits>
#include "Resources.hpp" 

template<typename Allocator>
concept Source = std::is_base_of_v<Resource, Allocator>;

template<typename Allocator>
concept Strategy = !std::is_base_of_v<Resource, Allocator>;

template<Source Resource, Strategy Allocator>
class Pool {
public: 

    Pool(Resource resource)
    :   resource_(resource)
    ,   allocator_(nullptr, 0) {}

    void reserve(std::size_t memory) {
        assert(size_ == 0 && "Cannot resize reserved memory");
        size_ = memory;
        buffer_ = resource_.allocate(size_);
        allocator_ = Allocator(buffer_, size_);
    } 

    Allocator const& allocator() const {
        return allocator_;
    }

    ~Pool() { 
        if (buffer_) resource_.deallocate(buffer_, size_);
    }
    
    Pool(const Pool&) = delete; 
    Pool& operator=(const Pool&) = delete;
     
    Pool(Pool&& other) noexcept
        : size_(std::exchange(other.size_, 0))
        , buffer_(std::exchange(other.buffer_, nullptr))
        , resource_(std::move(other.resource_))
        , allocator_(std::move(other.allocator_)) {}

        
    Pool& operator=(Pool&& other) noexcept {
        if (this != &other) {
            if (buffer_) {
                resource_.deallocate(buffer_, size_);
            }

            size_ = std::exchange(other.size_, 0);
            buffer_ = std::exchange(other.buffer_, nullptr);
            resource_ = std::move(other.resource_);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }

    
private:
    std::size_t size_ = 0;
    void* buffer_ = nullptr;
    Resource resource_;
    Allocator allocator_;
};  

#endif // POOL_HPP