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

#ifndef SHAPE_HPP
#define SHAPE_HPP
 
/**
 * @file Shape.hpp
 * @author Eric Cardozo
 * @date 2025
 * @brief Defines the `tannic::Shape` class for representing tensor dimensions.   
 * 
 * This file is part of the Tannic Tensor Library and defines the `Shape` class,
 * a compact, constexpr-friendly abstraction for representing the shape (i.e., dimensions)
 * of tensors and expression-like objects. Shapes are integral to indexing, broadcasting,
 * and tensor manipulation throughout the library.
 *
 * The shape supports:
 * 
 * - Compile-time construction (constexpr) 
 * 
 * - Convenient initialization from iterators, lists, or variadic arguments
 * 
 * - Safe indexing, including Python-style negative indexing
 *
 * All expression-like objects in the library expose a `Shape` as part of their interface,
 * making this file central to the Tannic design.
 */   

#include <type_traits> 
#include <array>
#include <cstdint>
#include <cassert>  
#include <initializer_list>
#include <ostream>
 
#include "Concepts.hpp" 
#include "Indexing.hpp" 

namespace tannic { 
    
/**
 * @class Shape
 * @brief Represents the shape (dimensions) of an tensor-like expression.
 *
 * The `Shape` class provides a constexpr-friendly abstraction over an expression shape and
 * provides constructors and accessors for tensor shapes, including support for
 * initializer lists, iterators, and iterable containers.
 * 
 * Shapes are limited to a maximum rank of 8 dimensions. 
 *
 *
 * #### Example:
 * 
 * ```cpp
 * using namespace tannic;
 * 
 * constexpr Shape s1(2, 3, 4);
 * constexpr Shape s2(2, 3, 4);
 * static_assert(s1 == s2);
 * std::cout << s1 << std::endl; // Shape(2,3,4)
 * ```
 */
class Shape {
public:
    static constexpr uint8_t limit = 8;  

public:
    using rank_type = uint8_t;      ///< Type used for rank (number of dimensions).
    using size_type = size_t;  ///< Type used for size and shape dimensions. 
 
    /**
    * @brief Default constructor. Initializes a shape of rank 0 and size 1 (shape of scalars).
    */
    constexpr Shape() noexcept = default;


    /**
    * @brief Constructs a shape from an initializer list of dimension sizes.
    * @param shape An initializer list of dimension sizes.
    * @note Asserts if the number of dimensions exceeds `limit`.
    * 
    * #### Example:
    *
    *  ```cpp 
    * Shape shape({2, 3, 4});
    * std::cout << shape;  // Output: Shape(2, 3, 4)
    * ```
    */
    constexpr Shape(std::initializer_list<size_type> shape) {
        assert(shape.size() < limit && "Rank limit exceeded");
        rank_ = static_cast<size_type>(shape.size());
        size_type dimension = 0;
        for (auto size : shape) {
            sizes_[dimension++] = size;
        } 
        assert(rank_ <= limit && "Rank limit exceeded"); 
    } 


    /**
    * @brief Constructs a shape from a list of size arguments.
    *
    * @tparam Sizes Variadic list of integral size arguments.
    * @param sizes Sizes of each dimension.
    *
    * @note Asserts if the number of dimensions exceeds `limit`.
    *   
    * #### Example:
    * 
    * ```cpp 
    * Shape shape(2, 3, 4);
    * std::cout << shape;  // Output: Shape(2, 3, 4)
    * ```
    */
    template<Integral... Sizes>
    constexpr Shape(Sizes... sizes) 
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {      
        assert(rank_ <= limit && "Rank limit exceeded"); 
    }


    /**
     * @brief Constructs a shape from any iterable container of sizes.
     * @tparam Sizes An iterable with begin() and end() methods.
     * @param sizes Container of dimension sizes.
     */
    template<Iterable Sizes>
    constexpr Shape(Sizes&& sizes) {
        size_type dimension = 0;
        for (auto size : sizes) {
            sizes_[dimension++] = static_cast<size_type>(size);
        }
        rank_ = dimension; 
        assert(rank_ <= limit && "Rank limit exceeded"); 
    } 

 
    /**
     * @brief Constructs a shape from a pair of iterators.
     *
     * @tparam Iterator Type of the iterator (must satisfy the Iterator concept).
     * @param begin Iterator to the beginning of dimension sizes.
     * @param end Iterator to the end of dimension sizes.
     *
     * @note Asserts if the number of dimensions exceeds `limit`.
     *
     * #### Example:
     * 
     * ```cpp 
     * std::vector<std::size_t> dims = {2, 3, 4};
     * Shape shape(dims.begin(), dims.end());
     * std::cout << shape;  // Output: Shape(2, 3, 4)
     * ```
     */
    template<Iterator Iterator>
    constexpr Shape(Iterator begin, Iterator end) {
        size_type dimension = 0;
        for (Iterator iterator = begin; iterator != end; ++iterator) { 
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension;  
        assert(rank_ <= limit && "Rank limit exceeded"); 
    }

public:    
    /**
     * @brief Returns a pointer to the internal size data (non-const).
     * @return Pointer to the first element of the sizes array.
     */
    constexpr size_type* address() noexcept {
        return sizes_.data();
    }

    /**
     * @brief Returns a pointer to the internal size data (const).
     * @return Const pointer to the first element of the sizes array.
     */
    constexpr size_type const* address() const noexcept {
        return sizes_.data();
    }
    
    /**
     * @brief Returns the number of dimensions (rank).
     * @return Rank of the shape.
     */
    constexpr rank_type rank() const noexcept { 
        return rank_; 
    }
     

    /// @name Iterators
    /// @{
 
    constexpr auto begin() { 
        return sizes_.begin(); 
    }
     
    constexpr auto end() { 
        return sizes_.begin() + rank_; 
    }
 
    constexpr auto begin() const { 
        return sizes_.begin(); 
    }  

    constexpr auto end() const { 
        return sizes_.begin() + rank_; 
    } 
 
    constexpr auto cbegin() const { 
        return sizes_.cbegin(); 
    }
     
    constexpr auto cend() const { 
        return sizes_.cbegin() + rank_; 
    }
    /// @}

    /**
     * @brief Returns the first dimension size.
     * @return Size of the first dimension.
     */
    constexpr auto front() const noexcept { 
        return sizes_.front(); 
    }

    /**
     * @brief Returns the last dimension size.
     * @return Size of the last dimension.
     * @note Asserts if shape is empty.
     */
    constexpr auto back() const { 
        assert(rank_ > 0 && "Cannot call back() on an empty Shape");
        return sizes_[rank_ - 1];
    }       

    /**
     * @brief Accesses a dimension by index (const).
     * @tparam Index Integral index type.
     * @param index Index of the dimension to access (supports negative indexing).
     * @return Size of the dimension at the given index.
     */
    template<Integral Index>
    constexpr auto const& operator[](Index index) const { 
        return sizes_[indexing::normalize(index, rank())]; 
    }

    /**
     * @brief Accesses a dimension by index (non-const).
     * @tparam Index Integral index type.
     * @param index Index of the dimension to access (supports negative indexing).
     * @return Reference to the dimension size at the given index.
     */
    template<Integral Index>
    constexpr auto& operator[](Index index) {
        return sizes_[indexing::normalize(index, rank())]; 
    } 


    /**
     * @brief Expands the shape's last dimension with a given size. Increments the rank by one. 
     * @param size The size of the dimension that will be added to the back of the shape. 
     */
    constexpr void expand(size_type size) { 
        assert(rank_ < limit && "Rank limit exceeded");  
        sizes_[rank_] = size;
        rank_ += 1; 
    }


private:
    rank_type rank_{0};      
    std::array<size_type, limit> sizes_{}; 
};
 
/**
 * @brief Equality comparison operator for shapes.
 * @param first First shape.
 * @param second Second shape.
 * @return True if both shapes have the same rank and sizes, false otherwise.
 */
constexpr bool operator==(Shape const& first, Shape const& second) {
    if (first.rank() != second.rank()) return false;
    for (Shape::size_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
}  

inline std::ostream& operator<<(std::ostream& os, Shape const& shape) {
    os << "Shape(";
    for (Shape::rank_type dimension = 0; dimension < shape.rank(); ++dimension) {
        os << static_cast<unsigned int>(shape[dimension]);
        if (dimension + 1 < shape.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
} 


} // namespace tannic

#endif // SHAPE_HPP