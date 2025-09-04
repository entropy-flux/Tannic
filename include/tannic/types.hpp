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
 
#ifndef TYPES_HPP
#define TYPES_HPP
 
/**
 * @file types.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Core type system for the Tannic Tensor Library.
 * 
 * Defines the fundamental data types supported by tensors and provides utilities for: 
 * 
 * ### Supported Data Types
 * 
 * The `type` enum defines all supported numeric types:
 * ```cpp
 * enum type {
 *     none,       // Invalid type 
 *     boolean     // 1-bit-packed boolean (true, false).
 *     int8,       // 8-bit integer
 *     int16,      // 16-bit integer
 *     int32,      // 32-bit integer
 *     int64,      // 64-bit integer
 *     float16,    // 16-bit float
 *     float32,    // 32-bit float
 *     float64,    // 64-bit float (double)
 *     complex64,  // 64-bit complex (two float32)
 *     complex128, // 128-bit complex (two float64)
 *     TYPES       // Count of types
 * };
 * ```
 *
 * ### Key Functions
 * 
 * - `dsizeof(type)`: Returns byte size of a type
 * 
 * - `dnameof(type)`: Returns string name of a type
 * 
 * - `dcodeof(type)`: Returns numeric code for serialization
 * 
 * - `dtypeof(code)`: Converts code back to type enum
 * 
 * - `operator<<`: Stream output support
 *  
 * ### Example
 * 
 * std::cout << dnameof(float32); // "float32"
 * std::cout << dsizeof(complex128); // 16
 */


#include <iostream>
#include <cstdint>  
#include <string>   
#include <complex>
#include "runtime/types.h"

namespace tannic {  

struct boolean_t {
    bool value = false;
    boolean_t() : value(false) {}
    template<typename T> boolean_t(T value) : value(static_cast<bool>(value)) {}
    operator bool() const { return value; } 
    void write(std::byte*, std::ptrdiff_t) const;  
};  
 
struct float16_t {
    uint16_t bits = 0; 
    float16_t() : bits(0) {}
    template<std::integral T>
    float16_t(T value) : float16_t(static_cast<float>(value)) {}
    float16_t(double value) : float16_t(static_cast<float>(value)) {}
    float16_t(float); 
    operator float() const;
}; 

struct bfloat16_t {
    uint16_t bits = 0; 
    bfloat16_t() : bits(0) {}
    template<std::integral T>
    bfloat16_t(T value) : bfloat16_t(static_cast<float>(value)) {}
    bfloat16_t(double value) : bfloat16_t(static_cast<float>(value)) {}
    bfloat16_t(float); 
    operator float() const;
};
  
/**
 * @brief Returns the size in bytes of a given tensor data type.
 * @param type The data type to query
 * @return Size of the type in bytes (0 for `none`)
 * 
 * @note 
 * - `boolean` is stored as bit-packed (1 bit per element). Since the size is
 *   less than one byte, this function returns 0. To compute the actual storage
 *   requirement for N elements, use `(N + 7) / 8` bytes.
 * - `complex64`  returns 8 (2 × float32).
 * - `complex128` returns 16 (2 × float64).
 * 
 * #### Example:
 * ```cpp
 * dsizeof(float32);  // returns 4
 * dsizeof(complex64); // returns 8
 * ```
 */
constexpr inline std::size_t dsizeof(type type) {
    switch (type) { 
        case boolean:   return 0;
        case int8:      return sizeof(int8_t);
        case int16:     return sizeof(int16_t);
        case int32:     return sizeof(int32_t);
        case int64:     return sizeof(int64_t);
        case float16:   return sizeof(float) / 2;
        case bfloat16:  return sizeof(float) / 2;
        case float32:   return sizeof(float);
        case float64:   return 2 * sizeof(float);
        case complex64: return 2 * sizeof(float);     
        case complex128:return 2 * sizeof(double);  
        default:        return 0;
    }
} 

/**
 * @brief Returns the total number of bytes required to store `nelements` elements
 *        of the given data type.
 *
 * @param type The data type.
 * @param nelements Number of elements.
 * @return Total size in bytes required.
 *
 * @note
 * - For `boolean`, storage is bit-packed. The result is `(nelements + 7) / 8`.
 * - For all other types, result is `dsizeof(type) * nelements`.
 *
 * #### Example:
 * ```cpp
 * nbytesof(float32, 10);   // returns 40
 * nbytesof(complex64, 5);  // returns 40
 * nbytesof(boolean, 100);  // returns 13
 * ```
 */
constexpr inline std::size_t nbytesof(type dtype, std::size_t nelements) {
    if (dtype == boolean) {
        return (nelements + 7) / 8; 
    }
    return dsizeof(dtype) * nelements;
}


/**
 * @brief Returns the string name of a given tensor data type.
 * @param type The data type to query
 * @return Human-readable type name ("none" for invalid types)
 * 
 * #### Example:
 * ```cpp
 * std::cout << dnameof(int32) << std::endl   // prints "int32"
 * std::cout << dnameof(complex128); // prints "complex128"
 * ```
 */
constexpr inline std::string dnameof(type type) {
    switch (type) { 
        case boolean:    return "boolean";
        case int8:       return "int8";
        case int16:      return "int16";
        case int32:      return "int32";
        case int64:      return "int64";
        case float16:    return "float16";
        case bfloat16:    return "bfloat16";
        case float32:    return "float32";
        case float64:    return "float64";
        case complex64:  return "complex64";
        case complex128: return "complex128";
        default:         return "none";
    }
} 

/**
 * @brief Returns the numeric code used for serialization of a data type.
 * @param type The data type to query
 * @return Unique numeric code (0 for `none`)
 * 
 * @note Code values follow simple pattern:
 * 
 * - Integers: 10-19 (integer dtypes)
 * 
 * - Floats: 20-29 (floating point dtypes)
 * 
 * - Complex: 30-39 (complex) 
 * 
 * While this grouping is intentional, it's not strictly enforced. When adding new types:
 * 
 * 1. Maintain this pattern where possible
 * 
 * 2. Document any deviations
 * 
 * 3. Keep codes unique across all types
 */
constexpr inline uint8_t dcodeof(type type) {
    switch (type) { 
        case boolean:   return 1;
        case int8:      return 12;
        case int16:     return 13;
        case int32:     return 14;
        case int64:     return 15;
        case float16:   return 23;
        case bfloat16:  return 231;
        case float32:   return 24;
        case float64:   return 25;
        case complex64: return 35;     
        case complex128:return 36;  
        default:        return 0;
    }
}

/**
 * @brief Converts a numeric type code back to its corresponding type enum. Used for
 * deserialization.
 * @param code The numeric type code to convert (as returned by dcodeof())
 * @return Corresponding type enum value (none for invalid codes)
 * 
 * @note This is the inverse operation of dcodeof(). The code values follow the same pattern.
 *  
 * @see dcodeof() for the reverse conversion
 * 
 */
constexpr inline type dtypeof(uint8_t code) {
    switch (code) {
        case 1 : return boolean;
        case 12: return int8;
        case 13: return int16;
        case 14: return int32;
        case 15: return int64;
        case 23: return float16;
        case 231:return bfloat16;
        case 24: return float32;
        case 25: return float64;
        case 35: return complex64;
        case 36: return complex128;
        default: return unknown;
    }
} 
  
template <typename T>
constexpr inline type dtypeof() {
    if constexpr (std::is_same_v<T, bool>) return boolean;
    else if constexpr (std::is_same_v<T, int8_t>)  return int8;
    else if constexpr (std::is_same_v<T, int16_t>) return int16;
    else if constexpr (std::is_same_v<T, int32_t>) return int32;
    else if constexpr (std::is_same_v<T, int64_t>) return int64;
    else if constexpr (std::is_same_v<T, float16_t>)   return float16;
    else if constexpr (std::is_same_v<T, bfloat16_t>)  return bfloat16;
    else if constexpr (std::is_same_v<T, float>)   return float32;
    else if constexpr (std::is_same_v<T, double>)  return float64;
    else if constexpr (std::is_same_v<T, std::complex<float>>)  return complex64;
    else if constexpr (std::is_same_v<T, std::complex<double>>) return complex128;
    else                                           return unknown;
}

inline std::ostream& operator<<(std::ostream& ostream, type type) {
    return ostream << dnameof(type);
} 

} // namespace tannic

#endif // TYPES_HPP