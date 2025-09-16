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
 * @file views.hpp 
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
 * // View from (2, 3) to (3, 2)
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

#include "types.hpp"
#include "traits.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "concepts.hpp"
#include "exceptions.hpp"

namespace tannic {

class Tensor;

} namespace tannic::expression {   
   
/**
 * @class View
 * @brief Expression template for viewing a tensor with a new shape.
 *
 * @tparam Source The expression or tensor type being reshaped.
 *
 * The `View` view changes how the elements of a tensor are indexed
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
template<Composable Source>
class Reshape {
public:    
    /**
     * @brief Construct a reshaped view of the source tensor.
     *
     * @tparam Indexes Integral dimension sizes of the new shape.
     * @param source Reference to the source expression or tensor.
     * @param sizes Dimension sizes for the reshaped view.
     *
     * @throws Assertion failure if the number of elements in the new shape
     *         does not match the number of elements in the source.
     */ 
  
    template<Integral... Sizes>  
    constexpr Reshape(typename Trait<Source>::Reference source, Sizes... sizes)
    :   source_(source)
    { 
        std::array<long long, sizeof...(Sizes)> requested{ static_cast<long long>(sizes)... };
  
        std::size_t nelements = 1;
        std::size_t stride = 1;
        for (auto dimension = 0; dimension < source.shape().rank(); ++dimension) {
            auto index = source.shape().rank() - 1 - dimension; 
            if (source.strides()[index] != stride) {
                throw Exception("Only contiguous tensors allowed in view");
            }
            nelements *= source.shape()[dimension];
            stride *= source.shape()[index];
        }
        
        int inferred = -1;
        std::size_t accumulated = 1; 
        for (auto dimension = 0; dimension < requested.size(); ++dimension) { 
            auto index = requested.size() - 1 - dimension; 

            if (requested[dimension] == -1) {
                if (inferred != -1) throw Exception("Only one dimension can be inferred (-1) in view");
                inferred = dimension;
            } 
            
            else if (requested[dimension] < 0) {
                throw Exception("Invalid negative dimension in view");
            } 
            
            else {
                accumulated *= requested[dimension];
            }
            shape_.expand(requested[dimension]);
            strides_.expand(1);
        }
 
        if (inferred != -1) {
            if (nelements % accumulated != 0) throw Exception("Cannot infer dimension: source elements not divisible");
            shape_[inferred] = nelements / accumulated; 
        } 

        else if (accumulated != nelements) {
            throw Exception("Shape mismatch: view must preserve total number of elements");
        }

        for (auto dimension = shape_.rank() - 2; dimension >= 0; --dimension) {
            strides_[dimension] = strides_[dimension + 1] * shape_[dimension + 1];
        } 
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
template<Composable Source>
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
template<Composable Source, Integral ... Indexes>
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
    :   source_(source)
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
 * @class Expansion
 * @brief Expression template for expanding (broadcasting) singleton dimensions of a tensor.
 *
 * @tparam Source The expression or tensor type being expanded.
 *
 * The `Expansion` view allows a tensor to be broadcast along dimensions where the original
 * size is 1, enabling operations with larger tensors without copying data. Only dimensions
 * with size 1 in the source tensor can be expanded; other dimensions must match the target shape.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {1, 3});
 * auto Y = expand(X, 4, 3); // shape becomes (4, 3)
 * ```
 */
template<Composable Source>
class Expansion {
public:
    /**
     * @brief Construct an expansion view.
     *
     * Stores reference to the source tensor and the requested target sizes. 
     *
     * @tparam Sizes Integral dimension sizes of the target shape.
     * @param source Reference to the source tensor or expression.
     * @param sizes Dimension sizes for the expanded view.
     */    
    template<Integral... Sizes>
    constexpr Expansion(typename Trait<Source>::Reference source, Sizes... sizes)
    :   source_(source) {
        std::array<long long, sizeof...(Sizes)> requested{ static_cast<long long>(sizes)... }; 
 
        if (requested.size() < source.shape().rank()) 
            throw Exception("Expansion target rank must be >= source rank");

        std::size_t offset = requested.size() - source.shape().rank();
        for (std::size_t dimension = 0; dimension < requested.size(); ++dimension) {
            long long index = requested[dimension];
            std::size_t target;

            if (index == -1) {
                if (dimension < offset) {
                    throw Exception("Cannot use -1 for new leading dimensions");
                }
                target = source.shape()[dimension - offset];
            } else if (index <= 0) {
                throw Exception("Expansion size must be positive or -1");
            } else {
                target = static_cast<std::size_t>(index);
            }
 
            if (dimension < offset) { 
                shape_.expand(target);
                strides_.expand(0);
            } 
            
            else { 
                if (source.shape()[dimension - offset] == 1 && target > 1) {
                    shape_.expand(target);
                    strides_.expand(0);  // broadcast
                } else if (source.shape()[dimension - offset] == target) {
                    shape_.expand(target);
                    strides_.expand(source.strides()[dimension - offset]);
                } else {
                    throw Exception("Expansion only allows -1 (keep) or broadcasting singleton dims");
                }
            }
        }
    }


    /**
     * @return The data type of the tensor elements.
     *
     * Broadcasting does not change the element type, so the dtype
     * is forwarded from the source tensor.
     */
    constexpr type dtype() const { 
        return source_.dtype(); 
    }

    /**
     * @return The shape of the expanded view.
     *
     * Calculation:
     * - Returns the requested target shape stored in the constructor.
     * - Validates that non-singleton dimensions of the source match the requested size.
     *
     * @throws Exception if a non-singleton dimension in the source does not match the target.
     */
    constexpr Shape const& shape() const { 
        return shape_; 
    }


    /**
     * @return The strides of the expanded view.
     *
     * Calculation:
     * - Copy the source strides.
     * - For each dimension where the source size is 1 and target size > 1, set stride to 0.
     *   This ensures the same memory element is repeated along expanded dimensions.
     */
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    /**
     * @return The byte offset of the expanded view from the start of the storage.
     *
     * The offset is the same as the source tensor, because broadcasting
     * does not change the underlying data location.
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
 * @class Squeeze
 * @brief Expression template for removing singleton dimensions from a tensor.
 *
 * @tparam Source The expression or tensor type being squeezed.
 *
 * The `Squeeze` view removes dimensions of size 1 from the shape of a tensor.
 * This changes only the tensor metadata (shape and strides), not the underlying storage.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {1, 3, 1});
 * auto Y = squeeze(X); // shape becomes (3)
 * ```
 */
template<Composable Source>
class Squeeze {
public:
    /**
     * @brief Construct a squeezed view of the source tensor.
     *
     * @param source Reference to the source expression or tensor.
     *
     * This constructor removes all dimensions of size 1 from the source shape.
     * Strides corresponding to those singleton dimensions are also removed.
     */
    constexpr Squeeze(typename Trait<Source>::Reference source)
    : source_(source) {
        for (auto dimension = 0; dimension < source.shape().rank(); ++dimension) {
            if (source.shape()[dimension] != 1) {
                shape_.expand(source.shape()[dimension]);
                strides_.expand(source.strides()[dimension]);
            }
        }
    }

    /**
     * @return The runtime data type of the tensor elements.
     *
     * This is forwarded directly from the source expression.
     * Squeezing does not alter the tensor’s dtype.
     */
    constexpr type dtype() const { 
        return source_.dtype(); 
    }

    /**
     * @return The shape of the squeezed tensor.
     *
     * Calculation:
     * - Copies the source shape.
     * - Removes all dimensions of size 1.
     *
     * Example:
     * - Source shape: (1, 3, 1) → Squeezed shape: (3)
     */
    constexpr Shape const& shape() const { 
        return shape_; 
    }

    /**
     * @return The strides of the squeezed tensor.
     *
     * Calculation:
     * - Copies the source strides.
     * - Removes strides corresponding to dimensions of size 1.
     *
     * Example:
     * - Source strides: (3, 1, 1) with shape (1, 3, 1)
     * - Result after squeeze: (1) with shape (3)
     */
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    /**
     * @return The byte offset of the squeezed tensor from the start of storage.
     *
     * This is forwarded from the source tensor since squeezing
     * does not change the starting position of the data.
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
 * @class Unsqueeze
 * @brief Expression template for inserting singleton dimensions into a tensor.
 *
 * @tparam Source The expression or tensor type being unsqueezed.
 *
 * The `Unsqueeze` view adds new dimensions of size 1 at specified axes.
 * This changes only the tensor metadata (shape and strides), not the underlying storage.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {3});
 * auto Y = unsqueeze(X, 0);   // shape becomes (1, 3)
 * auto Z = unsqueeze(X, -1);  // shape becomes (3, 1)
 * ```
 */
template<Composable Source>
class Unsqueeze {
public:
    /**
     * @brief Construct an unsqueezed view of the source tensor.
     *
     * @tparam Axes Integral indices where new dimensions of size 1 should be inserted.
     * @param source Reference to the source expression or tensor.
     * @param axes One or more dimension indices (negative indices allowed).
     *
     * Example:
     * - `unsqueeze(X, 0)` inserts a new axis before the first dimension.
     * - `unsqueeze(X, -1)` inserts a new axis after the last dimension.
     *
     * @throws Exception if any normalized axis index is invalid.
     */
    template<Integral... Axes>
    constexpr Unsqueeze(typename Trait<Source>::Reference source, Axes... axes)
    : source_(source) {
        auto rank = source.shape().rank();
        std::vector<std::size_t> normalized{ static_cast<std::size_t>(indexing::normalize(axes, rank + sizeof...(axes)))... };
        std::sort(normalized.begin(), normalized.end());

        size_t dimensions = rank + normalized.size();
        size_t index = 0;
        size_t axis = 0;

        for (auto dimension = 0; dimension < dimensions; ++dimension) {
            if (axis < normalized.size() && dimension == normalized[axis]) {
                shape_.expand(1);
                strides_.expand( (index < source.strides().rank()) ? source.strides()[index] : 1 );
                ++axis;
            } else {
                shape_.expand(source.shape()[index]);
                strides_.expand(source.strides()[index]);
                ++index;
            }
        }
    }

    /**
     * @return The runtime data type of the tensor elements.
     *
     * This is forwarded directly from the source expression.
     * Unsqueezing does not alter the tensor’s dtype.
     */
    constexpr type dtype() const { 
        return source_.dtype(); 
    }

    /**
     * @return The shape of the unsqueezed tensor.
     *
     * Calculation:
     * - Copies the source shape.
     * - Inserts size-1 dimensions at the specified axes.
     *
     * Example:
     * - Source shape: (3)
     * - unsqueeze(X, 0) → shape: (1, 3)
     */
    constexpr Shape const& shape() const { 
        return shape_; 
    }

    /**
     * @return The strides of the unsqueezed tensor.
     *
     * Calculation:
     * - Copies the source strides.
     * - Inserts strides for new singleton dimensions.
     *   These strides are set to repeat the same memory element along the axis.
     */
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    /**
     * @return The byte offset of the unsqueezed tensor from the start of storage.
     *
     * This is forwarded from the source tensor since unsqueezing
     * does not change the starting position of the data.
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
 * @class Flatten
 * @brief Expression template for flattening a contiguous range of dimensions.
 *
 * @tparam Source The expression or tensor type being flattened.
 *
 * The `Flatten` view collapses dimensions between `start_dim` and `end_dim`
 * into a single dimension. This operation only modifies tensor metadata
 * (shape and strides) and does not move data.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = flatten(X, 1, -1); // shape becomes (2, 12)
 * auto Z = flatten(X);        // shape becomes (24)
 * ```
 */
template<Composable Source>
class Flatten {
public:
    constexpr Flatten(typename Trait<Source>::Reference source, int start = 0, int end = -1)
    :   source_(source) {
        auto rank = source.shape().rank();
 
        start = indexing::normalize(start, rank);
        end   = indexing::normalize(end, rank);

        if (start > end) {
            throw Exception("Flatten requires start_dim <= end_dim");
        }
 
        for (int dimension = 0; dimension < start; ++dimension) {
            shape_.expand(source.shape()[dimension]);
            strides_.expand(source.strides()[dimension]);
        }
 
        std::size_t flattened = 1;
        for (int dimension = start; dimension <= end; ++dimension) {
            flattened *= source.shape()[dimension];
        }
        shape_.expand(flattened);
        strides_.expand(source.strides()[end]);  
 
        for (int dimension = end + 1; dimension < rank; ++dimension) {
            shape_.expand(source.shape()[dimension]);
            strides_.expand(source.strides()[dimension]);
        }
    }

    /**
     * @return The runtime data type of the tensor elements.
     *
     * This is forwarded directly from the source expression.
     * Flatten does not alter the tensor’s dtype.
     */
    constexpr type dtype() const { return source_.dtype(); }

        /**
     * @return The shape of the flattened tensor.
     *
     * Calculation:
     * - Dimensions before `start` are copied unchanged.
     * - Dimensions between `start` and `end` (inclusive) are collapsed
     *   into a single dimension equal to the product of their sizes.
     * - Dimensions after `end` are copied unchanged.
     *
     * Example:
     * - Source shape: (2, 3, 4), start_dim = 1, end_dim = -1
     * - Flattened shape: (2, 12)
     */
    constexpr Shape const& shape() const { 
        return shape_; 
    }

    /**
     * @return The strides of the flattened tensor.
     *
     * Calculation:
     * - Strides before `start` are copied unchanged.
     * - The collapsed dimension is assigned the stride of the
     *   *last* flattened dimension (`end`).  
     *   This ensures contiguous indexing across the merged block.
     * - Strides after `end_dim` are copied unchanged.
     *
     * Example:
     * - Source strides: (12, 4, 1) with shape (2, 3, 4)
     * - Flattened (start=1, end=-1) → strides: (12, 1) with shape (2, 12)
     */
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    /**
     * @return The byte offset of the unsqueezed tensor from the start of storage.
     *
     * This is forwarded from the source tensor since flatten
     * does not change the starting position of the data.
     */
    std::ptrdiff_t offset() const { return source_.offset(); }

    Tensor forward() const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;
};

 

  
/*
----------------------------------------------------------------------------------------------------
*/


/**
 * @brief Creates a reshaped view of a tensor or expression.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Indexes New shape dimensions (integral values).
 * @param source The source expression.
 * @param indexes Dimension sizes for the new shape.
 * @return A `View` view expression.
 */
template<Composable Source, Integral ... Indexes>
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
template<Composable Source>
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
template<Composable Source, Integral ... Indexes>
constexpr auto permute(Source&& source, Indexes... indexes) {
    return Permutation<Source, Indexes...>(
        std::forward<Source>(source), 
        std::make_tuple(indexes...)
    );
}

  
/**
 * @brief Creates an expanded view of a tensor, broadcasting singleton dimensions.
 *
 * This function returns an `Expansion` expression that allows a tensor to be 
 * “expanded” along dimensions of size 1 without copying data. Expansion is only 
 * allowed along singleton dimensions; other dimensions must match the requested size.
 *
 * @tparam Source The tensor or expression type to expand.
 * @tparam Sizes Integral dimension sizes for the expanded view.
 * @param source The source tensor or expression.
 * @param sizes The target shape for the expanded view.
 * @return An `Expansion` object representing the broadcasted view.
 *
 * @throws Exception if:
 *   - The number of dimensions does not match the source rank.
 *   - A non-singleton dimension in the source does not match the requested size.
 *
 * Example usage:
 * ```cpp
 * Tensor X(float32, {1, 3}); // shape: (1, 3)
 * auto Y = expand(X, 4, 3);  // shape: (4, 3), broadcasts along the first dimension
 *
 * std::cout << Y.shape() << std::endl;   // prints (4, 3)
 * std::cout << Y.strides() << std::endl; // prints (0, original_stride[1])
 * ```
 */
template<Composable Source, Integral... Sizes>
constexpr auto expand(Source&& source, Sizes... sizes) {
    return Expansion<Source>(std::forward<Source>(source), sizes...);
}


/**
 * @brief Removes all singleton dimensions from a tensor (squeeze).
 *
 * This function returns a `Squeeze` expression that reinterprets the
 * source tensor without its size-1 dimensions.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source tensor or expression.
 * @return A `Squeeze` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {1, 3, 1});
 * auto Y = squeeze(X); // shape: (3)
 * ```
 */
template<Composable Source>
constexpr auto squeeze(Source&& source) {
    return Squeeze<Source>(std::forward<Source>(source));
}


/**
 * @brief Inserts singleton dimensions at the specified axes (unsqueeze).
 *
 * This function returns an `Unsqueeze` expression that reinterprets
 * the source tensor with new dimensions of size 1 added.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Axes One or more integral indices where new dimensions will be inserted.
 * @param source The source tensor or expression.
 * @param axes Axis indices (negative indices allowed).
 * @return An `Unsqueeze` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {3});
 * auto Y = unsqueeze(X, 0);  // shape: (1, 3)
 * auto Z = unsqueeze(X, -1); // shape: (3, 1)
 * ```
 */
template<Composable Source, Integral... Axes>
constexpr auto unsqueeze(Source&& source, Axes... axes) {
    return Unsqueeze<Source>(std::forward<Source>(source), axes...);
}


/**
 * @brief Flattens dimensions of a tensor into a single dimension.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source tensor or expression.
 * @param start_dim First dimension to flatten (default = 0).
 * @param end_dim Last dimension to flatten (default = -1, meaning last dim).
 * @return A `Flatten` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = flatten(X, 1, -1); // shape: (2, 12)
 * auto Z = flatten(X);        // shape: (24)
 * ```
 */
template<Composable Source>
constexpr auto flatten(Source&& source, int start = 0, int end   = -1) {
    return Flatten<Source>(std::forward<Source>(source), start, end);
}


} namespace tannic {

using expression::view;
using expression::transpose;
using expression::permute;
using expression::expand;
using expression::squeeze;
using expression::unsqueeze;
using expression::flatten;

} // namespace tannic

#endif // VIEWS_HPP