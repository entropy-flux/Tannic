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

#ifndef RESOURCES_HPP
#define RESOURCES_HPP

#include <cstddef>
#include <vector>
#include <span>

enum unit : std::size_t {
    B = 1,
    KB = 1024,
    MB = 1024 * 1024,
    GB = 1024 * 1024 * 1024,
    UNIT
};

struct Resource{};

struct Host : Resource {  
    void* allocate(std::size_t memory) const { return ::operator new(memory); }
    void deallocate(void* address, std::size_t size) const { ::operator delete(address); }
    unsigned long long available() const;
};

struct Device : Resource { 
    Device(int id) : id(id) {}
    int id;
    void* allocate(std::size_t memory);               
    void deallocate(void* address, std::size_t size); 
    unsigned long long available() const;
};

class Resources {
public:
    Resources();
    static Host host() { return Host{}; } 
    std::span<const Device> devices() const {
        return std::span<const Device>(devices_.data(), devices_.size());
    }

private:
    std::vector<Device> devices_;
};
 

#endif // RESOURCES_HPP
