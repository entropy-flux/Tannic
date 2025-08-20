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
 
#ifndef VIEWS_HPP
#define VIEWS_HPP 

/**
 * @file Views.hpp 
 * @author Eric Hermosis
 * @date 2025  
 * @brief Implements views for tensors in the Tannic Tensor Library.
 * 
 * This header defines expression templates for tensors views
 * without copying data. These views operate on the underlying tensor metadata
 * (shape, strides, and offset) to reinterpret how elements are accessed while
 * preserving the original storage.
 *
 * Example usage:
 *
 * ```cpp
 * Tensor X(float32, {2, 3}); X.initialize();
 * 
 * // Reshape from (2, 3) to (3, 2)
 * auto Y = view(X, 3, 2);
 * std::cout << Y.shape() << std::endl; // (3, 2)
 *
 * // Swap the first and second dimensions
 * auto Z = transpose(X, 0, 1);
 * std::cout << Z.shape() << std::endl; // (3, 2)
 * ```
 */

#include <utility> 
#include <algorithm>
#include <numeric>
#include <vector>

#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
#include "Concepts.hpp"
#include "Exceptions.hpp"

namespace tannic {

class Tensor;

namespace expression {   
   
/**
 * @class Reshape
 * @brief Expression template for viewing a tensor with a new shape.
 *
 * @tparam Source The expression or tensor type being reshaped.
 *
 * The `Reshape` view changes how the elements of a tensor are indexed
 * by updating its shape and recomputing its strides without moving data.
 *
 * This operation requires that the total number of elements in the new
 * shape matches the original tensor's total number of elements.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3});
 * auto Y = view(X, 3, 2); // new shape: (3, 2)
 * ```
 */
template<Expression Source>
class Reshape {
public:  

    /**
     * @brief Construct a reshaped view of the source tensor.
     *
     * @tparam Indexes Integral dimension sizes of the new shape.
     * @param source Reference to the source expression or tensor.
     * @param indexes Dimension sizes for the reshaped view.
     *
     * @throws Assertion failure if the number of elements in the new shape
     *         does not match the number of elements in the source.
     */
    template<Integral... Indexes>  
    constexpr Reshape(Trait<Source>::Reference source, Indexes... indexes)
    :   shape_(indexes...) 
    ,   strides_(shape_)
    ,   source_(source) {
        std::size_t elements = 0;
        for (std::size_t dimension = 0; dimension < sizeof...(indexes); ++dimension) {
            elements += strides_[dimension] * (shape_[dimension] - 1);
        } 
        elements += 1;  
        if (elements != std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{} ))
            throw Exception("Shape mismatch: view must preserve total number of elements");
    }

    /**
     * @return The runtime data type of the tensor elements.
     *
     * This simply forwards the call to the underlying source expression’s
     * `dtype`. Since a view (reshape or transpose) does not alter
     * the actual stored values.
     */
    constexpr type dtype() const {
        return source_.dtype();
    }

    /**
     * @return The shape of the reshaped view.
     *
     * In a reshape, this value is explicitly provided in the constructor
     * through `indexes...` and stored in `shape_`.  
     */
    constexpr Shape const& shape() const {
        return shape_;
    }

    /**
     * @return The strides of the reshaped view.
     *
     * For a reshape, strides are recomputed from the new `shape_` so that
     * the element layout in memory matches the original data ordering. 
     */
    constexpr Strides const& strides() const {
        return strides_;
    }

    /**
     * @return The byte offset of the reshaped view from the start of the storage.
     *
     * This value is forwarded from the source expression’s `offset()` method.
     * The offset represents the memory position (in bytes) where the first
     * element of the view begins relative to the start of the underlying
     * storage buffer. Since reshape do not move elements in
     * memory, the offset is unchanged from the source tensor.
     */
    std::ptrdiff_t offset() const {
        return source_.offset();
    }
 
    Tensor forward() const;

private:
    Shape shape_;
    Strides strides_;
    typename Trait<Source>::Reference source_;                
};   


/**
 * @class Transpose
 * @brief Expression template for transposing two dimensions of a tensor.
 *
 * @tparam Source The expression or tensor type being transposed.
 *
 * The `Transpose` view swaps the shape and strides of two dimensions
 * without moving data. This is useful for reordering axes for operations
 * like matrix multiplication.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3});
 * auto Y = transpose(X, 0, 1); // shape becomes (3, 2)
 * auto Z = X.transpose(0, 1); // oop syntax.
 * ```
 */
template<Expression Source>
class Transpose {
public: 

    /**
     * @brief Construct a transposed view of the source tensor.
     *
     * @param source Reference to the source expression or tensor.
     * @param dimensions A pair of dimension indices to swap.
     */
    constexpr Transpose(typename Trait<Source>::Reference source, std::pair<int, int> dimensions)
    :   shape_(source.shape()) 
    ,   strides_(source.strides())
    ,   source_(source)
    ,   dimensions_(dimensions) {  
        auto rank = source.shape().rank();    
        std::swap(shape_[indexing::normalize(dimensions.first, rank)], shape_[indexing::normalize(dimensions.second, rank)]); 
        std::swap(strides_[indexing::normalize(dimensions.first, rank)], strides_[indexing::normalize(dimensions.second, rank)]);   
    }

    /**
     * @return The data type of the tensor elements.
     *
     * Transposing does not change the element type. This value is
     * returned from the source expression's `dtype()`.
     */
    constexpr type dtype() const {
        return source_.dtype();
    } 
        
    /**
     * @return The shape of the transposed view.
     *
     * The shape is initially copied from the source tensor, then
     * in the constructor the two specified dimensions are swapped.
     */
    constexpr Shape const& shape() const {
        return shape_;
    } 

    /**
     * @return The strides of the transposed view.
     *
     * Strides are copied from the source tensor in the constructor,
     * and then the strides for the two swapped dimensions are exchanged
     * to match the new layout in memory.
     */
    constexpr Strides const& strides() const {
        return strides_;
    }  

    /**
     * @return The byte offset of the transposed view from the start of the storage.
     *
     * This is taken directly from the source expression’s `offset()` because
     * transposing only changes how dimensions are indexed, not where the
     * first element is stored.
     */
    std::ptrdiff_t offset() const {
        return source_.offset();
    }

    Tensor forward() const;

private:
    Shape shape_; 
    Strides strides_;
    typename Trait<Source>::Reference source_; 
    std::pair<int, int> dimensions_;
};   


/**
 * @class Permutation
 * @brief Expression template for reordering tensor dimensions according to a specified permutation.
 *
 * @tparam Source The expression or tensor type being permuted.
 * @tparam Indexes A parameter pack of integral indices defining the permutation order.
 *
 * The `Permutation` view reorders the dimensions of a tensor according to a
 * user-specified sequence of indices. This changes how elements are accessed
 * along each axis, but does not move the underlying data in memory.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = permute(X, 2, 0, 1); // shape becomes (4, 2, 3)
 * ```
 *
 * @note The number of indices in the permutation must match the rank of the tensor.
 *       Otherwise, an exception is thrown.
 */
template<Expression Source, Integral ... Indexes>
class Permutation {
public:
    /**
     * @brief Constructs a permuted view of the source tensor.
     *
     * @param source Reference to the source expression or tensor.
     * @param indexes A tuple of dimension indices specifying the new order.
     *
     * @throws Exception if the number of indices does not match the source tensor's rank.
     *
     * This constructor normalizes the indices to ensure they are valid for the tensor's rank,
     * and then expands the `shape_` and `strides_` of the permuted view accordingly.
     */
    constexpr Permutation(typename Trait<Source>::Reference source, std::tuple<Indexes...> indexes) 
        : source_(source)
    {
        if (sizeof...(Indexes) != source_.shape().rank()) {
            throw Exception("Permutation rank must equal tensor rank");
        }

        std::apply([&](auto... indexes) {
            (([&]{
                int dimension = indexing::normalize(indexes, source_.shape().rank());
                shape_.expand(source_.shape()[dimension]);
                strides_.expand(source_.strides()[dimension]);
            }()), ...);
        }, indexes);
    }

    /**
     * @return The data type of the tensor elements.
     *
     * The element type remains the same as the source tensor.
     */
    constexpr type dtype() const { 
        return source_.dtype(); 
    }

    /**
     * @return The shape of the permuted view.
     *
     * The shape is reordered according to the permutation indices provided
     * in the constructor.
     */
    constexpr Shape const& shape() const { 
        return shape_; 
    }

    /**
     * @return The strides of the permuted view.
     *
     * Strides are reordered to match the permuted axes, ensuring that
     * element access corresponds to the new dimension order.
     */
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    /**
     * @return The byte offset of the permuted view from the start of the storage.
     *
     * This forwards directly from the source tensor. The data buffer is not
     * moved, so the offset remains the same.
     */
    std::ptrdiff_t offset() const { 
        return source_.offset(); 
    } 

    Tensor forward() const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;              
};



/**
 * @brief Creates a reshaped view of a tensor or expression.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Indexes New shape dimensions (integral values).
 * @param source The source expression.
 * @param indexes Dimension sizes for the new shape.
 * @return A `Reshape` view expression.
 */
template<Expression Source, Integral ... Indexes>
constexpr auto view(Source&& source, Indexes ... indexes) {
    return Reshape<Source>(
        std::forward<Source>(source), indexes...
    );
} 


/**
 * @brief Creates a transposed view of a tensor or expression by swapping two dimensions.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source expression.
 * @param first First dimension index to swap.
 * @param second Second dimension index to swap.
 * @return A `Transpose` view expression.
 */
template<Expression Source>
constexpr auto transpose(Source&& source, int first, int second) {
    return Transpose<Source>(
        std::forward<Source>(source),
        std::make_pair(first, second)
    );
} 

/**
 * @brief Creates a permuted view of a tensor or expression.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Indexes Integral indices specifying the permutation order.
 * @param source The source expression.
 * @param indexes Sequence of dimension indices indicating the new axis order.
 * @return A `Permutation` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = permute(X, 2, 0, 1); // shape becomes (4, 2, 3)
 * ```
 */
template<Expression Source, Integral ... Indexes>
constexpr auto permute(Source&& source, Indexes... indexes) {
    return Permutation<Source, Indexes...>(
        std::forward<Source>(source), 
        std::make_tuple(indexes...)
    );
}
  

} // namespace expression

using expression::view;
using expression::transpose;
using expression::permute;

} // namespace tannic

#endif // VIEWS_HPP