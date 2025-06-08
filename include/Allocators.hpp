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

#ifndef ALLOCATORS_HPP
#define ALLOCATORS_HPP

#include <variant>  
#include <cstddef>
#include <cstring>
#include "Resources.hpp" 
#include "Pool.hpp"
#include "Freelist.hpp"

struct View {
    void* buffer = nullptr;
    void* allocate(std::size_t) { return buffer; }
    void deallocate(void*, std::size_t) { buffer = nullptr; }
    void copy(void* address, void const* value, std::size_t size) const { std::memcpy(address, value, size); }
    bool compare(void const* address, void const* value, std::size_t size) const { return std::memcmp(address, value, size) == 0; }
};

using Allocator = std::variant<Device, Host, View>;

#endif // ALLOCATORS_HPP