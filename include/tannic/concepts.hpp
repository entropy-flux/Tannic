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

#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP 

/**
 * @file concepts.hpp
 * @brief Defines core C++20 concepts used throughout the Tannic Tensor Library.
 * @date 2025
 * @brief This header the `tannic::Expression` fundamental concept as a compile time
 * interface for all operations defined in this library and a few other concepts used
 * internally to validate template arguments in compile-time expressions. 
 * 
 * @note Unless you are extending the library with your custom expressions you can skip this
 * secction.
 */  
 
#include <concepts>
#include <iterator>   
#include "types.hpp"  

namespace tannic {

class Shape;
class Strides;
class Scalar;
class Tensor;
class Context; 

template<typename T>
concept Composable = std::same_as<T, Scalar> || ( requires(const T expression) {
        { expression.dtype()   } -> std::same_as<type>;
        { expression.shape()   } -> std::same_as<Shape const&>;
        { expression.strides() } -> std::same_as<Strides const&>;
        { expression.offset()  };
        { expression.forward(std::declval<Context const&>()) };
    }
);

template<typename T>
concept Operator = requires(T operation, const Shape& first, const Shape& second) {
    { T::promote(type{}, type{}) } -> std::same_as<type>;
    { T::broadcast(first, second) } -> std::same_as<Shape>; 
};

template<typename Function>
concept Functional = requires(Function function, const Tensor& input, Tensor& output) {
    { function(input, output) } -> std::same_as<void>;
};


template<typename T>
concept Iterable = requires(T type) { std::begin(type); std::end(type); };   

template <typename T>
concept Iterator = std::input_iterator<T>;  

template<class T>
concept Integral = std::integral<std::remove_cvref_t<T>>;

template<typename T>
concept Arithmetic =
    std::is_arithmetic_v<T> ||
    std::same_as<T, float16_t> ||
    std::same_as<T, bfloat16_t> ||
    std::same_as<T, std::complex<float>> ||
    std::same_as<T, std::complex<double>>;
    
template<typename T>
concept Assignable = requires(T t, const std::byte* ptr, std::ptrdiff_t offset) {
    { t.assign(ptr, offset) } -> std::same_as<void>;
};

template<typename T>
concept Comparable = requires(const T t, const std::byte* ptr, std::ptrdiff_t offset) {
    { t.compare(ptr, offset) } -> std::same_as<bool>;
}; 
 
} // namespace tannic

#endif // CONCEPTS_HPP