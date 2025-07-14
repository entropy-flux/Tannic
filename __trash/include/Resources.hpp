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

#ifndef RESOURCES_HPP
#define RESOURCES_HPP

#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>
#include <variant>
#include "core/resources.h"

class Host {
public: 
    Host();
    void* allocate(std::size_t nbytes) const;
    void deallocate(void* address, std::size_t nbytes) const;
    int id() const { return -1; }
    resource kind() const { return HOST; } 
};
 
class Device {
public: 
    Device(int id = 0);
    void* allocate(std::size_t nbytes) const;
    void deallocate(void* address, std::size_t nbytes) const; 
    int id() const;
    resource kind() const { return DEVICE; }

private:
    int id_ = 0;
};

using Allocator = std::variant<Host, Device>;

#endif // RESOURCES_HPP