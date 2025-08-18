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

#ifndef RESOURCES_HPP
#define RESOURCES_HPP 

/**
 * @file Resources.hpp
 * @author Eric Hermosis  
 * @date 2025
 * @brief Memory resource management for heterogeneous computing. 
 *
 * Defines the abstract memory architecture:
 * - Host: The primary execution environment and memory space (eg. CPU)
 * - Device: Secondary execution environments with separate memory spaces (eg. GPU)
 * - Explicit memory management interface for each domain
 *
 * The model enforces:
 * 1. Separate memory address spaces
 * 2. Distinct allocation semantics per domain  
 * 3. Explicit data transfer requirements 
 * 
 * @warning This interface is under active development and may undergo 
 *          breaking changes in future releases. The host/device abstraction
 *          is particularly prone to API evolution as hardware capabilities advanc
 */

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>
#include <variant>   

namespace tannic {

/**
 * @brief Host memory domain
 * 
 * - Primary memory space attached to the root execution environment
 * - Manages the default address space for the main processor  
 * - Required for all operations in the root execution context 
 * 
 * @warning This class interface is experimental and likely to change as 
 * the backend evolves.
 */
class Host {
public: 

    /**
     * @brief Constructs a host memory allocator  
     * @post Defaults to pageable allocation strategy
     */
    Host() = default;
 
    /**
     * @brief Allocates memory in host address space
     * @param nbytes Contiguous memory block size in bytes
     * @return Pointer to allocated memory in host space
     * @pre nbytes > 0
     * @post Returned pointer is valid in host execution context
     */
    void* allocate(std::size_t nbytes) const;

    /**
     * @brief Releases host memory resources
     * @param address Base pointer of allocation
     * @param nbytes Size of original allocation (for accounting)
     * @pre address points to valid host allocation
     * @post Memory is returned to host allocator
     */
    void deallocate(void* address, std::size_t nbytes) const; 

    /**
     * @brief Domain identifier constant
     * @return Constant -1 representing the host domain
     */
    int id() const { return -1; } 

    bool pageable() const {
        return pageable_;
    }

    bool pinned() const {
        return pinned_;
    }

private:
    bool pageable_ = true; 
    bool pinned_ = false;
};  


/**
 * @brief Device enumeration singleton
 * 
 * - Maintains global knowledge of available compute domains  
 * - Provides immutable count of attached devices
 * - Follows Meyer's singleton pattern for thread-safe initialization
 * - Exists as a single instance per process
 * 
 *
 * @warning Device memory management is highly likely to change as the
 * backend evolves.
 */

class Devices {
private:
    int count_ = 0; 
    Devices();

public: 
    Devices(const Devices&) = delete;
    Devices& operator=(const Devices&) = delete;
    Devices(Devices&&) = delete;
    Devices& operator=(Devices&&) = delete; 
    ~Devices() = default;

    static Devices& instance() {
        static Devices instance;
        return instance;
    }

    static int count() {
        Devices& devices = instance();
        return devices.count_;
    } 
}; 


/**
 * @brief Device memory domain
 * 
 * - Secondary memory space attached to a compute accelerator
 * - Requires explicit data transfer from host domain  
 * - Allocation strategies may vary by device capabilities
 * - Memory operations are asynchronous by default
 * - Each instance manages a single logical device 
 * 
 * @warning Device memory management is highly likely to change as the
 * backend evolves.
 */
class Device {
public: 

    /**
     * @brief Constructs a device memory allocator
     * @param id Target device identifier (0-based)
     * @param blocking Enable synchronous operation mode
     */
    Device(int id = 0, bool blocking = false); 

    /**
     * @brief Allocates device memory  
     * @param nbytes Contiguous memory block size in bytes
     * @return Device pointer valid in target domain  
     */
    void* allocate(std::size_t nbytes) const;


    /**
     * @brief Releases device memory
     * @param address Device pointer from allocate()
     * @param nbytes Size of original allocation
     */
    void deallocate(void* address, std::size_t nbytes) const; 

    /**
     * @brief Device identifier
     * @return Non-negative device index
     */
    int id() const noexcept { 
        return id_; 
    }
    
    bool blocking() const {
        return blocking_;
    }

private:
    int id_ = 0;
    bool blocking_ = false;
};
 

/**
 * @brief Memory allocator variant type.
 *
 * Type-safe union of host and device memory allocators. 
 */
using Allocator = std::variant<Host, Device>;

} // namespace tannic

#endif // RESOURCES_HPP