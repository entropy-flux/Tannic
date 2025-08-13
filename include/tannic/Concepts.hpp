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

#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP 

/**
 * @file Concepts.hpp
 * @brief Defines core C++20 concepts used throughout the Tannic Tensor Library.
 * @date 2025
 * @brief This header the `tannic::Expression` fundamental concept as a compile time
 * interface for all operations defined in this library and a few other concepts used
 * internally to validate template arguments in compile-time expressions. 
 */  
 
#include <concepts>
#include <iterator>   
#include "Types.hpp"

namespace tannic {

class Shape;
class Strides;
class Tensor; 

/**
 * @concept Expression
 * @brief 
 * Defines the core protocol for all expression-like types in the Tannic Tensor Library.
 *
 * @details
 * This concept specifies the interface required to participate in tensor operations such as
 * broadcasting, slicing, evaluation, and composition. Any type that satisfies this concept
 * is considered a valid expression—this includes tensors, views, lazy expressions, and future
 * compile-time tensor constructs.
 *
 * #### Requirements:
 * 
 * A type `T` satisfies `Expression` if it provides:
 * 
 * - `dtype()   -> tannic::type`                    — the data type of the expression.
 * 
 * - `shape()   -> tannic::Shape const&`            — the shape (dimensions) of the expression.
 * 
 * - `strides() -> tannic::Strides const&`          — the strides associated with the memory layout.
 * 
 * - `offset()  -> std::ptrdiff_t  `                — the memory offset from the buffer pointer.
 * 
 * - `forward() -> tannic::Tensor `                 — the tensor that will express the computation.
 *
 * The methods `dtype()`, `shape()`, and `strides()` **can be `constexpr`**.
 * All major components—such as `Tensor`, views, and expression trees—must satisfy this concept.
 * This ensures a consistent interface for operations like `+`, `matmul`, `broadcast`, etc.
 *
 * The `Expression` concept is designed to be **extensible**: custom user-defined types can
 * participate in the tensor system by conforming to this protocol, enabling interoperability
 * with native Tannic expressions.
 *
 * #### Example:
 * 
 * ```
 * template<tannic::Expression E>
 * void print_metadata(E const& expr) {
 *     std::cout << expr.shape() << " | dtype = " << expr.dtype() << '\n';
 * }
 * ```
 */
template<typename T>
concept Expression = requires(const T expression) {
    { expression.dtype()   } -> std::same_as<type>;
    { expression.shape()   } -> std::same_as<Shape const&>;
    { expression.strides() } -> std::same_as<Strides const&>;
      expression.offset();
      expression.forward();
}; 

/**
 * @brief Concept defining requirements for tensor operation types
 *  
 * This concept specifies the interface that all tensor operators must implement:
 * - Type promotion rules between operands
 * - Shape broadcasting behavior
 *
 * A type satisfies the Operator concept if it provides:
 * 1. A static `promote` method that determines the result type
 *    from two input types
 * 2. A static `broadcast` method that computes the output shape
 *    from two input shapes 
 * 
 * @see Operations.hpp (tensor operations)
 */
template<typename T>
concept Operator = requires(T operation, const Shape& first, const Shape& second) {
    { T::promote(type{}, type{}) } -> std::same_as<type>;
    { T::broadcast(first, second) } -> std::same_as<Shape>; 
};

/**
 * @brief Concept for unary mathematical function operations 
 *
 * Specifies requirements for types that represent mathematical function operations:
 * - Must provide a call operator that performs element-wise transformation 
 *
 * Used to constrain template parameters for mathematical function operations.
 */
template<typename Function>
concept Functional = requires(Function function, const Tensor& input, Tensor& output) {
    { function(input, output) } -> std::same_as<void>;
};

/**
 * @concept Iterable
 * @brief Requires a type to be iterable via `std::begin` and `std::end`. 
 */
template<typename T>
concept Iterable = requires(T type) { std::begin(type); std::end(type); };   

/**
 * @concept Iterator
 * @brief Requires a type to satisfy the C++20 `std::input_iterator` concept. 
 */
template <typename T>
concept Iterator = std::input_iterator<T>;  

/**
 * @concept Integral
 * @brief Requires a type to be an integral type (e.g., `int`, `std::size_t`). 
 */
template<typename T>
concept Integral = std::is_integral_v<T>; 

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