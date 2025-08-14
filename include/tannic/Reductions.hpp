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
    ,   strides_(shape_) {}

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
        Tensor source = operand.forward();
        Tensor result(dtype(), shape(), strides(), offset());          
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
        assert(shape.rank() >= 1); 
        Shape reduced;
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension != static_cast<size_t>(axis))
                reduced.expand(shape[dimension]);
        }
        return reduced;
    }
 
    /**
     * @brief Performs the argmax operation
     * @param input Source tensor
     * @param output Result tensor containing indices
     */
    void forward(Tensor const& input, Tensor& output) const;
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
        assert(shape.rank() >= 1); 
        Shape reduced; 
        for (uint8_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension != static_cast<uint8_t>(axis))
                reduced.expand(shape[dimension]);
        }
        return reduced;
    } 

    /**
     * @brief Performs the argmin operation
     * @param input Source tensor
     * @param output Result tensor containing indices
     */
    void forward(Tensor const&, Tensor&) const; 
};


/**
 * @brief Reduction sum operation.
 *
 * Computes the sum of elements along the specified axis.
 * Maintains input dtype but removes the reduced dimension.
 */
struct Argsum {
    int axis;
    bool keepdim = false;  ///< Whether to keep the reduced dimension
    
    constexpr type reduce(type dtype) const {
        return dtype;  // Sum preserves dtype
    }

    constexpr Shape reduce(Shape const& shape) const {
        Shape reduced;
        for (size_t dim = 0; dim < shape.rank(); ++dim) {
            if (dim != static_cast<size_t>(axis)) {
                reduced.expand(shape[dim]);
            } else if (keepdim) {
                reduced.expand(1);  // Keep reduced dim as size 1
            }
        }
        return reduced;
    } 

    void forward(Tensor const& input, Tensor& output) const;
};


/**
 * @brief Reduction mean operation.
 *
 * Computes the mean of elements along the specified axis.
 * Converts integer inputs to float and removes the reduced dimension.
 */
struct Argmean {
    int axis;
    bool keepdim = false;
    
    constexpr type reduce(type dtype) const { 
        assert(dtype == float32 | dtype == float64 && "Integral dtypes not supported.");
        return dtype;
    }

    constexpr Shape reduce(Shape const& shape) const {
        Shape reduced;
        for (size_t dim = 0; dim < shape.rank(); ++dim) {
            if (dim != static_cast<size_t>(axis)) {
                reduced.expand(shape[dim]);
            } else if (keepdim) {
                reduced.expand(1);
            }
        }
        return reduced;
    } 

    void forward(Tensor const& input, Tensor& output) const;
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
    return Reduction<Argmin, Source>{ 
        {indexing::normalize(axis, source.shape().rank())}, std::forward<Source>(source) 
    };
} 

template<Expression Source>
constexpr auto sum(Source&& source, int axis = -1, bool keepdim = false) {
    return Reduction<Argsum, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim},
        std::forward<Source>(source)
    };
}

template<Expression Source>
constexpr auto mean(Source&& source, int axis = -1, bool keepdim = false) {
    return Reduction<Argmean, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim},
        std::forward<Source>(source)
    };
}

 
} // namespace expression
 
using expression::argmax;
using expression::argmin;
using expression::sum;
using expression::mean;

} // namespace tannic

#endif // REDUCTIONS_HPP