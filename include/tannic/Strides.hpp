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

#ifndef STRIDES_HPP
#define STRIDES_HPP 

/**
 * @file Strides.hpp
 * @author Eric Cardozo
 * @date 2025
 * @brief Memory layout specification for tensor dimensions in the Tannic Tensor Library. 
 * 
 * Defines `tannic::Strides`, which determines how elements are spaced in memory
 * across tensor dimensions. Key features:
 * 
 * - **Memory layout control**: Row-major (default), column-major, or custom striding.
 * 
 * - **Constexpr support**: Compile-time stride computation.
 * 
 * - **Shape integration**: Auto-computes contiguous strides from `Shape`.
 * 
 * - **Negative indexing**: Python-style dimension access  
 * 
 * All expression-like objects in the library expose an `Strides` as part of their interface,
 * making this file central to the Tannic design.
 *
 * @see Shape.hpp (tensor dimensions), Indexing.hpp (negative index handling) 
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <ostream>
 
#include "Concepts.hpp"
#include "Shape.hpp" 
#include "Indexing.hpp"
#include "Exceptions.hpp"

namespace tannic {
 
/**
 * @class Strides
 * @brief Represents the memory strides associated with a tensor shape.
 *
 * `Strides` describes how many elements must be skipped in memory to move between
 * elements along each tensor dimension. It is a key part of defining how multidimensional
 * data is stored and accessed in memory.
 *
 * A `Strides` object can be constructed manually (via variadic arguments or iterators),
 * or automatically from a `Shape` (default row-major/contiguous layout).
 *
 * Maximum rank is limited to `Strides::limit = 8` for constexpr compatibility and static layout. This limit
 * may be removed in C++26 when std::vector become constexpr friendly.
 *
 * #### Example (Contiguous Layout):
 * 
 * ```cpp
 * using namespace tannic;
 *
 * Shape shape({2, 3, 4});
 * Strides strides(shape);
 *
 * std::cout << strides;    // Output: Strides(12, 4, 1)
 * ```
 *
 * #### Example (Custom Strides):
 * 
 * ```cpp
 * Strides strides(6, 2, 1); // Custom strides for a specific memory layout
 * ```
 */
class Strides {
public:  
    static constexpr uint8_t limit = 8;  

public:
    using rank_type = uint8_t;      ///< Type used for rank (number of dimensions).
    using size_type = int64_t;  ///< Type used for size and shape dimensions.
 
 
    /// @brief Default constructor (rank 0).
    constexpr Strides() noexcept = default; 
  
    /**
     * @brief Constructs strides from variadic size arguments.
     * @tparam Sizes Variadic list of integral stride values.
     * @param sizes Strides for each dimension.
     * 
     * @note Asserts if the number of dimensions (rank) exceeds `limit`.
     * 
     * #### Example:
     * 
     * ```cpp
     * constexpr tannic::Strides s(6, 2, 1);
     * std::cout << s;  // Output: Strides(6, 2, 1)
     * ```
     */
    template<Integral... Sizes>
    constexpr Strides(Sizes... sizes)
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {
        if (rank_ > limit) 
            throw Exception("Strides rank limit exceeded"); 
    } 

    /**
     * @brief Constructs strides from a pair of iterators.
     *
     * @tparam Iterator Type of the iterator (must satisfy the Iterator concept).
     * @param begin Iterator to the beginning of stride values.
     * @param end Iterator to the end of stride values.
     * 
     * @note Asserts if the number of dimensions exceeds `limit`.
     *
     * #### Example:
     * 
     * ```cpp
     * std::vector<std::size_t> values = {12, 4, 1};
     * tannic::Strides s(values.begin(), values.end());
     * ```
     */
    template<Iterator Iterator>
    constexpr Strides(Iterator begin, Iterator end) { 
        size_type dimension = 0;
        for (auto iterator = begin; iterator != end; ++iterator) {
            assert(dimension < limit && "Strides rank limit exceeded");
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension;
        if (rank_ > limit) 
            throw Exception("Strides rank limit exceeded"); 
    } 
    
    /**
     * @brief Constructs strides from a shape assuming row-major layout.
     * @param shape Shape from which to compute contiguous strides.
     *
     * #### Example:
     * 
     * ```cpp
     * constexpr tannic::Shape shape(2, 3, 4);
     * constexpr tannic::Strides s(shape);
     * std::cout << s;  // Output: Strides(12, 4, 1)
     * ```
     */
    constexpr Strides(const Shape& shape) { 
        rank_ = shape.rank();
        if (rank_ == 0) return;
        
        sizes_[rank_ - 1] = 1;
        for (int size = rank_ - 2; size >= 0; --size) {
            sizes_[size] = sizes_[size + 1] * shape[size + 1];
        }
        if (rank_ > limit) 
            throw Exception("Strides rank limit exceeded"); 
    }

public:
    /**
     * @brief Returns a pointer to the underlying data (non-const).
     * @return Pointer to beginning of stride values.
     */
    constexpr size_type* address() noexcept {
        return sizes_.data();
    }

    /**
     * @brief Returns a pointer to the underlying data (const).
     * @return Pointer to beginning of stride values.
     */
    constexpr size_type const* address() const noexcept {
        return sizes_.data();
    }

    /**
     * @brief Returns the number of dimensions (rank).
     * @return Rank of the strides.
     */
    constexpr auto rank() const noexcept { 
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
     * @brief Returns the first stride value.
     * @return Stride of the first dimension.
     */
    constexpr auto front() const { 
        return sizes_.front(); 
    }

    /**
     * @brief Returns the last stride value.
     * @return Stride of the last dimension.
     */
    constexpr auto back() const {
        return sizes_[rank_];
    }
  
    /**
     * @brief Accesses a stride value by index (const).
     * @tparam Index Integral index type.
     * @param index Index of the stride to access (supports negative indexing).
     * @return Value of the stride at the given index.
     */
    template<Integral Index>
    constexpr auto const& operator[](Index index) const { 
        return sizes_[indexing::normalize(index, rank())]; 
    }

    /**
     * @brief Accesses a stride value by index (non-const).
     * @tparam Index Integral index type.
     * @param index Index of the stride to access (supports negative indexing).
     * @return Reference to the stride value at the given index.
     */
    template<Integral Index>
    constexpr auto& operator[](Index index) {
        return sizes_[indexing::normalize(index, rank())]; 
    }  
 
    /**
     * @brief Expands the strides's last dimension with a given size. Increments the rank by one. 
     * @param size The size of the dimension that will be added to the back of the strides. 
     */
    constexpr void expand(size_type size) { 
        if (rank_ + 1 > limit) 
            throw Exception("Strides rank limit exceeded"); 
        sizes_[rank_] = size;
        rank_ += 1; 
    } 
    
private:
    rank_type rank_{0};
    std::array<size_type, limit> sizes_{};
};

/**
 * @brief Equality comparison for strides.
 * @return True if all stride sizes are equal for each dimension.
 */
constexpr bool operator==(Strides const& first, Strides const& second) {
    if (first.rank() != second.rank()) return false;
    for (Strides::rank_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
}

inline std::ostream& operator<<(std::ostream& os, Strides const& strides) {
    os << "Strides(";
    for (Strides::rank_type dimension = 0; dimension < strides.rank(); ++dimension) {
        os << static_cast<unsigned int>(strides[dimension]);
        if (dimension + 1 < strides.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
} 

} // namespace tannic

#endif // STRIDES_HPP