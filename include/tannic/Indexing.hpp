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

#ifndef INDEXING_HPP
#define INDEXING_HPP

/**
 * @file Indexing.hpp 
 * @author Eric Hermosis
 * @date 2025  
 * @brief Utilities for index normalization and slicing ranges in the Tannic Tensor Library.
 *
 * This header defines helper structures and functions for working with tensor
 * indices and slices. The utilities are inspired by Python's negative indexing
 * and range semantics, providing a consistent and safe interface for 
 * normalizing indices into valid positive ranges within tensor dimensions. 
 */

#include <cassert> 
#include "Concepts.hpp"
#include "Exceptions.hpp"

namespace tannic::indexing {
  
/**
 * @struct Range
 * @brief Represents a half-open interval `[start, stop)` for slicing.
 *
 * @details
 * The `Range` struct mimics Python's slice behavior:
 * - `start` is inclusive
 * - `stop` is exclusive
 * - Negative values are allowed and are interpreted relative to the size of
 *   the dimension being indexed.
 *
 * By default:
 * - `start = 0` (beginning of the axis)
 * - `stop  = -1` (interpreted as "until the end" when normalized)
 *
 * @see normalize(Range, Size)
 */
struct Range {
    int start = 0;
    int stop = -1;
};

/**
 * @brief Normalize a possibly-negative index into the valid range `[0, bound)`.
 *
 * @tparam Index Integral type of the index
 * @tparam Size Integral type of the bound
 * @param index Index to normalize (may be negative, counting from the end)
 * @param bound Upper bound (dimension size)
 * @return Normalized non-negative index
 *
 * @throws Assertion failure if the normalized index is out of bounds
 *
 * @details
 * This function is used **internally** by the tensor library when processing
 * indexing operations. It allows Python-like negative indices, where `-1`
 * refers to the last element of a dimension, `-2` the second-to-last, etc.
 *
 * #### Example:
 * ```cpp
 * int i = tannic::indexing::normalize(-1, 5); // → 4
 * int j = tannic::indexing::normalize(2, 5);  // → 2
 * ```
 *
 * @note End-users typically don’t call this directly; it underpins higher-level
 * indexing (`tensor[i]`) and slicing operations.
 */
template<Integral Index, Integral Size>
constexpr inline Index normalize(Index index, Size bound) {
    if (index < 0) index += bound;
    if (index < 0 | index > bound)
        throw Exception("Index out of bounds");
    return index;
}  

/**
 * @brief Normalize a slicing range into valid `[start, stop)` indices.
 *
 * @tparam Size Integral type of the dimension size
 * @param range A possibly-negative `Range`
 * @param size  Size of the dimension being indexed
 * @return A new `Range` with non-negative, normalized indices
 *
 * @details
 * This function is used **internally** by the tensor library when converting
 * user-provided slicing ranges into valid `[start, stop)` intervals.
 * Negative `start` and `stop` values are supported and are interpreted relative
 * to the end of the dimension:
 * - `start = -1` → last element
 * - `stop  = -1` → one-past-the-end
 *
 * #### Example:
 * ```cpp
 * auto r1 = tannic::indexing::normalize({-3, -1}, 10);
 * // r1.start = 7, r1.stop = 10
 * ```
 *
 * @note End-users generally use `tannic::range{}` or slicing syntax;
 * this function ensures those inputs are mapped into safe, non-negative
 * indices before evaluation.
 */
template<Integral Size>
constexpr inline Range normalize(Range range, Size size) {
    int start = range.start < 0 ? size + range.start : range.start;
    int stop = range.stop < 0 ? size + range.stop + 1 : range.stop;

    if (start < 0 | start > size | stop < 0 | stop > size)
        throw Exception("Range out of bounds");
    return {start, stop};
}  
 
} // namespace TANNIC::indexing

namespace tannic {

/**
 * @typedef range
 * @brief Convenience alias for `tannic::indexing::Range`.
 *
 * This alias allows using `tannic::range` directly when declaring
 * slices without fully qualifying the `indexing::Range` type.
 */
using range = indexing::Range;
} 

#endif // INDEXING_H