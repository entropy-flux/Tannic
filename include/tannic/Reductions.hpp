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

#ifndef REDUCTIONS_HPP
#define REDUCTIONS_HPP 
 
/**
 * @file Reductions.hpp
 * @author Eric Cardozo
 * @date 2025
 * @brief Defines reduction operations for tensor expressions.
 *
 * This header provides reduction operations for tensor-like objects,
 * implemented as expression templates. Currently supports:
 * - argmax: Index of maximum value along axis
 * - argmin: Index of minimum value along axis
 *
 * All reductions are lazy-evaluated and maintain proper shape/dtype transformations.
 * Part of the Tannic Tensor Library.
 */
 
#include <array>
#include <vector>
#include <cassert>

#include "Concepts.hpp"
#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp" 
#include "Tensor.hpp" 
#include "Indexing.hpp"

namespace tannic { 

namespace expression {

/**
 * @brief Expression template for reduction operations.
 *
 * Represents a lazily evaluated reduction operation applied to a tensor expression.
 * Handles proper shape, stride and dtype transformations for the reduced result.
 *
 * @tparam Reducer Reduction operation type (must implement reduce() and forward())
 * @tparam Operand Input expression type satisfying the Expression concept
 */
template<class Reducer, Expression Operand>
class Reduction {
public:
    Reducer reducer;
    typename Trait<Operand>::Reference operand;

    /**
     * @brief Constructs a Reduction expression
     * @param reducer Reduction operation functor
     * @param operand Input tensor expression
     */
    constexpr Reduction(Reducer reducer, typename Trait<Operand>::Reference operand)
    :   reducer(reducer)
    ,   operand(operand)
    ,   dtype_(reducer.reduce(operand.dtype()))
    ,   shape_(reducer.reduce(operand.shape()))
    ,   strides_(reducer.reduce(operand.strides())) {}

    /**
     * @brief Returns the data type of the reduced result
     * @return Transformed dtype after reduction
     */
    constexpr type dtype() const {
        return dtype_;
    }

    /**
     * @brief Returns the shape of the reduced result
     * @return Reduced shape after applying the reduction
     */
    constexpr Shape const& shape() const {
        return shape_;
    }

    /**
     * @brief Returns the strides of the reduced result
     * @return Adjusted strides for the reduced shape
     */
    constexpr Strides const& strides() const {
        return strides_;
    }

    /**
     * @brief Returns the offset of the result tensor
     * @return Always 0 since reductions create new tensors
     */
    std::ptrdiff_t offset() const {
        return 0;
    }

    /**
     * @brief Evaluates the reduction expression
     * @return New Tensor containing the reduced result
     */
    Tensor forward() const {   
        Tensor source = source.forward();
        Tensor result(dtype_, shape_, strides_, offset());          
        reducer.forward(source, result);
        return result;
    }

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};

/**
 * @brief Reduction argmax operation.
 *
 * Finds the indices of maximum values along the specified axis.
 * Reduces input dtype to int64 and removes the reduced dimension.
 */
struct Argmax {   
    int axis;  ///< Axis along which to find maxima 
    
    /**
     * @brief Determines output dtype for argmax
     * @param dtype Input dtype
     * @return Always returns int64 for indices
     */
    constexpr type reduce(type dtype) const {
        return int64;
    }

    /**
     * @brief Computes reduced shape for argmax
     * @param shape Input shape
     * @return Shape with reduced dimension removed
     * @throws assert if input has rank < 1
     */
    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1 && "Argmax requires at least 1D tensor");
        return Shape(shape.begin(), shape.end()-1);
    }

    /**
     * @brief Computes reduced strides for argmax
     * @param strides Input strides
     * @return Strides adjusted for reduced shape
     */
    constexpr Strides reduce(Strides const& strides) const {
        if (strides.rank() == 1) {
            return Strides();
        }
        
        uint8_t rank = strides.rank() - 1; 
        std::vector<std::size_t> sizes(rank);
        for(uint8_t dimension = 0; dimension < rank - 2; ++dimension) {
            sizes[dimension] = strides[dimension];
        }
        sizes[rank - 1] = strides.back();
        return Strides(sizes.begin(), sizes.end());
    } 
    
    /**
     * @brief Performs the argmax operation
     * @param input Source tensor
     * @param output Result tensor containing indices
     */
    void forward(Tensor const&, Tensor&) const;
};

/**
 * @brief Reduction argmin operation
 *
 * Finds the indices of minimum values along the specified axis.
 * Reduces input dtype to int64 and removes the reduced dimension.
 */
struct Argmin {   
    int axis;  ///< Axis along which to find minima 

    /**
     * @brief Determines output dtype for argmin
     * @param dtype Input dtype
     * @return Always returns int64 for indices
     */
    constexpr type reduce(type dtype) const {
        return int64;
    }

    /**
     * @brief Computes reduced shape for argmin
     * @param shape Input shape
     * @return Shape with reduced dimension removed
     * @throws assert if input has rank < 1
     */
    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1 && "Argmax requires at least 1D tensor");
        return Shape(shape.begin(), shape.end()-1);
    }

    /**
     * @brief Computes output strides after reducing the specified axis
     * @param strides Input tensor strides before reduction
     * @return New strides with the reduced dimension removed
     *
     * @details When reducing along an axis (for argmax/argmin), the output tensor
     * loses that dimension. This requires adjusting the strides accordingly:
     * 
     * 1. For scalar results (rank-1 input reduced to rank-0):
     *    - Returns empty strides (scalar has no strides)
     *    
     * 2. For higher-rank tensors:
     *    - Copies all strides except the one corresponding to the reduced axis
     *    - Maintains the original memory layout for remaining dimensions
     *    - The last stride is preserved to maintain contiguous access 
     * 
     * ```
     * Input shape: [3,4,5], strides: [20,5,1] (C-contiguous)
     * After reducing axis=1 (shape becomes [3,5]):
     * Output strides: [20,1]  // Skips the reduced dimension's stride (5) 
     * ```
     */
    constexpr Strides reduce(Strides const& strides) const {
        if (strides.rank() == 1) {
            return Strides();
        }
        
        uint8_t rank = strides.rank() - 1; 
        std::vector<std::size_t> sizes(rank);
        for(uint8_t dimension = 0; dimension < rank - 2; ++dimension) {
            sizes[dimension] = strides[dimension];
        }
        sizes[rank - 1] = strides.back();
        return Strides(sizes.begin(), sizes.end());
    } 

    /**
     * @brief Performs the argmin operation
     * @param input Source tensor
     * @param output Result tensor containing indices
     */
    void forward(Tensor const&, Tensor&) const; 
};

/**
 * @brief Creates an argmax reduction expression
 * @tparam Source Input expression type
 * @param source Input tensor expression
 * @param axis Axis along which to find maxima (default: last dimension)
 * @return Reduction expression for argmax
 * @note Currently only supports axis=-1 (last dimension)
 */
template<Expression Source>
constexpr auto argmax(Source&& source, int axis = -1) { 
    assert(axis == -1 && "Axis different from default not working in argmax. Open a PR if you find the issue.");
    return Reduction<Argmax, Source>{
        {indexing::normalize(axis, source.shape().rank())}, std::forward<Source>(source) 
    };
}

/**
 * @brief Creates an argmin reduction expression
 * @tparam Source Input expression type
 * @param source Input tensor expression
 * @param axis Axis along which to find minima (default: last dimension)
 * @return Reduction expression for argmin
 * @note Currently only supports axis=-1 (last dimension)
 */
template<Expression Source>
constexpr auto argmin(Source&& source, int axis = -1) { 
    assert(axis == -1 && "Axis different from default not working in argmin. Open a PR if you find the issue.");
    return Reduction<Argmin, Source>{ 
        {indexing::normalize(axis, source.shape().rank())}, std::forward<Source>(source) 
    };
}
 
} // namespace expression

// Bring reduction functions into tannic namespace
using expression::argmax;
using expression::argmin;

} // namespace tannic

#endif // REDUCTIONS_HPP