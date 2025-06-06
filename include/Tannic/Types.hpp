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
    using retrieve_function = std::any (*)(void const*); 
    using assign_function = void(*)(void*, std::any const&);
    using compare_function = bool(*)(void const*, std::any const&);
    using print_function = std::ostream&(*)(std::ostream&, void const*); 

    const char* name;
    size_type size;
    retrieve_function retrieve;
    assign_function assign;
    compare_function compare;
    print_function print; 
};


template<typename T>
inline void assign(void* address, std::any const& value) {
    *reinterpret_cast<T*>(address) = std::any_cast<T>(value);
}


template<typename T>
inline bool compare(void const* address, std::any const& value) {
    return *reinterpret_cast<T const*>(address) == std::any_cast<T>(value);
}


template<typename T>
inline std::any retrieve(void const* address) {
    return *reinterpret_cast<T const*>(address); 
}


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
        .retrieve = nullptr,
        .assign = nullptr,
        .compare = nullptr,
        .print = nullptr 
    },

    
    [integer8] = {
        .name = "integer8",
        .size = sizeof(int8_t),
        .retrieve = retrieve<int8_t>,
        .assign = assign<int8_t>,
        .compare = compare<int8_t>,
        .print = print<int8_t>  
    },

    [integer16] = {
        .name = "integer16",
        .size = sizeof(int16_t),
        .retrieve = retrieve<int16_t>,
        .assign = assign<int16_t>,
        .compare = compare<int16_t>,
        .print = print<int16_t>
    },
    
    [integer32] = {
        .name = "integer32",
        .size = sizeof(int32_t),
        .retrieve = retrieve<int32_t>,
        .assign = assign<int32_t>,
        .compare = compare<int32_t>,
        .print = print<int32_t>
    },

    
    [integer64] = {
        .name = "integer64",
        .size = sizeof(int64_t),
        .retrieve = retrieve<int64_t>,
        .assign = assign<int64_t>,
        .compare = compare<int64_t>,
        .print = print<int64_t>
    },

    
    [float32] = {
        .name = "float32",
        .size = sizeof(float),
        .retrieve = retrieve<float>,
        .assign = assign<float>,
        .compare = compare<float>,
        .print = print<float>
    },

    
    [float64] = {
        .name = "float64",
        .size = sizeof(double),
        .retrieve = retrieve<double>,
        .assign = assign<double>,
        .compare = compare<double>,
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