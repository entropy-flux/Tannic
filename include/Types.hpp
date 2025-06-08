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

#ifndef TYPES_HPP
#define TYPES_HPP

#include <iostream>
#include <cstdint>
#include <stdexcept>    
#include <any>
#include <cassert>

enum type : uint8_t {
    any,
    integer8,
    integer16,
    integer32,
    integer64,
    float32,
    float64,
    TYPES
};

struct Traits {
    using size_type = size_t; 
    using print_function = std::ostream&(*)(std::ostream&, void const*); 

    const char* name;
    size_type size; 
    print_function print; 
};

 

template<typename T>
inline std::ostream& print(std::ostream& os, void const* address) {
    return os << *reinterpret_cast<T const*>(address);
}


template<>
inline std::ostream& print<int8_t>(std::ostream& os, void const* address) {
    return os << +(*reinterpret_cast<int8_t const*>(address));
}
 

static constexpr Traits traits[TYPES] = { 
    [any] = {
        .name = "any",
        .size = 0, 
        .print = nullptr 
    }, 
    
    [integer8] = {
        .name = "integer8",
        .size = sizeof(int8_t), 
        .print = print<int8_t>  
    },

    [integer16] = {
        .name = "integer16",
        .size = sizeof(int16_t), 
        .print = print<int16_t>
    },
    
    [integer32] = {
        .name = "integer32",
        .size = sizeof(int32_t), 
        .print = print<int32_t>
    },

    
    [integer64] = {
        .name = "integer64",
        .size = sizeof(int64_t), 
        .print = print<int64_t>
    },

    
    [float32] = {
        .name = "float32",
        .size = sizeof(float), 
        .print = print<float>
    },

    
    [float64] = {
        .name = "float64",
        .size = sizeof(double), 
        .print = print<double>
    },

};


template<typename T>
inline T dcast(std::any& retrieved) {
    return std::any_cast<T>(retrieved);
}


inline constexpr size_t dsizeof(type type) {
    return traits[type].size;
}


inline std::ostream& operator<<(std::ostream& os, const type type) {
    assert(type < TYPES && "Invalid type");
    os << traits[type].name;
    return os;
}


inline std::ostream& operator<<(std::ostream& os, uint8_t value) {
    return os << static_cast<unsigned int>(value);
}


constexpr type promote(type first, type second) {
    assert(first < TYPES && second < TYPES && "Invalid type");
    if (first != second) 
        throw std::runtime_error("Type promotion rules not implemented yet");
    return first;
} 


#endif // TYPES_HPP