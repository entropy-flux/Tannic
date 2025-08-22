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

#ifndef BUFFER_HPP
#define BUFFER_HPP

/**
 * @file Buffer.hpp
 * @author Eric Hermosis  
 * @date 2025
 * @brief Memory buffer management for tensor storage
 * 
 * Provides a basic memory buffer abstraction that:
 * 
 * - Manages allocation and ownership of raw memory
 * 
 * - Supports both host and device memory through allocators
 * 
 * - Enforces move-only semantics for clear ownership
 * 
 * - Tracks memory region size and location
 * 
 * @see Allocator.hpp (supported allocators)
 */
 
#include <cstddef>   
#include "resources.hpp" 

namespace tannic { 

/**
 * @brief Managed memory buffer with explicit ownership
 * @details
 * Wraps a contiguous memory region with:
 * 
 * - Size tracking
 * 
 * - Allocator awareness  
 * 
 * - Move semantics for ownership transfer
 * 
 * - Const-correct access methods
 */
class Buffer {
public:   

    /**
     * @brief Constructs a buffer with specified size and allocator
     * @param nbytes Size of memory region in bytes
     * @param allocator Allocator to use (defaults to Host)
     */
    Buffer(std::size_t nbytes, Allocator allocator = Host{}); 

    // Non-copyable to maintain clear ownership
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /**
     * @brief Move constructor transfers ownership
     * @param other Source buffer to move from
     */
    Buffer(Buffer&& other) noexcept; 
    
    /**
     * @brief Move assignment transfers ownership 
     * @param other Source buffer to move from
     * @return Reference to this buffer
     */
    Buffer& operator=(Buffer&& other) noexcept;

    /**
     * @brief Destructor releases owned memory
     */
    ~Buffer(); 

    /**
     * @brief Gets writable pointer to memory
     * @return Pointer to allocated memory
     */
    void* address();

    /**
     * @brief Gets read-only pointer to memory  
     * @return Const pointer to allocated memory
     */
    const void* address() const;

    /**
     * @brief Gets buffer size in bytes
     * @return Size of allocated memory region
     */
    std::size_t nbytes() const; 
    
    /**
     * @brief Gets the allocator used for this buffer
     * @return Reference to the allocator instance
     */
    Allocator const& allocator() const; 

private:  
    void* address_ = nullptr;
    std::size_t nbytes_ = 0;
    Allocator allocator_ = Host{};
};

} // namespace TANNIC

#endif // BUFFER_HPP