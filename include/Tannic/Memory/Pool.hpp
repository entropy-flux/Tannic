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

#ifndef POOL_HPP
#define POOL_HPP
#include <type_traits>
#include "Memory/Resources.hpp"
#include "Memory/View.hpp"

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
    
    Pool(Pool&&) = delete;
    Pool& operator=(Pool&&) = delete;
    
private:
    std::size_t size_ = 0;
    void* buffer_ = nullptr;
    Resource resource_;
    Allocator allocator_;
};  

#endif // POOL_HPP