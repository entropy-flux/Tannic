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
#include <string>  
#include "core/types.h"

constexpr inline std::size_t dsizeof(type type) {
    switch (type) { 
        case int8:      return sizeof(int8_t);
        case int16:     return sizeof(int16_t);
        case int32:     return sizeof(int32_t);
        case int64:     return sizeof(int64_t);
        case float32:   return sizeof(float);
        case float64:   return sizeof(double);
        case complex64: return 2 * sizeof(float);     
        case complex128:return 2 * sizeof(double);  
        default:        return 0;
    }
}

constexpr inline std::string dnameof(type type) {
    switch (type) { 
        case int8:       return "int8";
        case int16:      return "int16";
        case int32:      return "int32";
        case int64:      return "int64";
        case float32:    return "float32";
        case float64:    return "float64";
        case complex64:  return "complex64";
        case complex128: return "complex128";
        default:         return "any";
    }
}
 
constexpr type complex(type dtype) {
    switch (dtype) {
        case float32: return complex64;
        case float64: return complex128;
        default: 
            throw std::invalid_argument("Only float32 and float64 can be converted to complex");
    }
}

constexpr type real(type dtype) {
    switch (dtype) {
        case complex64:  return float32;
        case complex128: return float64;
        default:         return dtype;  
    }
}

inline std::ostream& operator<<(std::ostream& ostream, type type) {
    return ostream << dnameof(type);
}

template<typename T>
struct Trait {
    using Reference = T;
}; 

#endif // TYPES_HPP