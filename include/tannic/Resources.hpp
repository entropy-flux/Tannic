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


#ifndef RESOURCES_HPP
#define RESOURCES_HPP

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>
#include <variant>  
#include "runtime/resources.h"

namespace tannic {

class Host {
public: 
    Host() = default;
    void* allocate(std::size_t nbytes) const;
    void deallocate(void* address, std::size_t nbytes) const;
    int id() const { return -1; } 

    bool pinned() const {
        return pinned_;
    }

private:
    bool pinned_ = false;
};
 
class Device {
public: 
    Device(int id = 0);
    void* allocate(std::size_t nbytes) const;
    void deallocate(void* address, std::size_t nbytes) const; 
    int id() const noexcept { return id_; }

private:
    int id_ = 0;
};
 
using Allocator = std::variant<Host, Device>;

} // namespace tannic

#endif // RESOURCES_HPP