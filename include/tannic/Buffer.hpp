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

#ifndef BUFFER_HPP
#define BUFFER_HPP
 
#include <cstddef>   
#include "Resources.hpp" 

namespace tannic {

class Buffer {
public:  
    Buffer(std::size_t nbytes, Allocator allocator = Host{}); 
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();
    void* address();
    const void* address() const;
    std::size_t nbytes() const;
    Allocator const& allocator() const; 

private:  
    void* address_ = nullptr;
    std::size_t nbytes_ = 0;
    Allocator allocator_ = Host{};
};

} // namespace TANNIC

#endif // BUFFER_HPP