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
 * - argmax: Index of maximum value along axis optionally keeping the original rank. 
 * - argmin: Index of minimum value along axis optionally keeping the original rank. 
 * - sum:  Sum of the values along axis optionally keeping the original rank. 
 * - mean: Mean of the values along axis optionally keeping the original rank. 
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
 * @brief Lazy reduction expression.
 *
 * Represents reductions like `sum`, `max`, or `mean` that collapse tensors along an axis.
 * Output shape/dtype are determined by the `Reducer`'s rules.
 *
 * @tparam Reducer Policy defining the reduction (e.g., `Argmax`).
 * @tparam Operand Input satisfying the `Expression` concept..
 */
template<class Reducer, Expression Operand>
class Reduction {
public:
    Reducer reducer;
    typename Trait<Operand>::Reference operand;
 
    constexpr Reduction(Reducer reducer, typename Trait<Operand>::Reference operand)
    :   reducer(reducer)
    ,   operand(operand)
    ,   dtype_(reducer.reduce(operand.dtype()))
    ,   shape_(reducer.reduce(operand.shape()))
    ,   strides_(shape_) {}
 
    constexpr type dtype() const {
        return dtype_;
    }
 
    constexpr Shape const& shape() const {
        return shape_;
    } 

    constexpr Strides const& strides() const {
        return strides_;
    }
 
    std::ptrdiff_t offset() const {
        return 0;
    }
 
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
 * @brief Finds the **indices of maximum values** along an axis.
 *
 * Output dtype is always `int64`. The reduced axis is removed by default (`keepdim=false`).
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{3, 1, 4}, 
 *             {1, 5, 9}};  // shape(2, 3)
 *
 * Tensor Y = argmax(X, 0);  // Reduce axis 0 (rows)
 * std::cout << Y << std::endl;
 * // Y = [0, 1, 1]          // Indexes of max values per column
 *
 * Tensor Z = argmax(X, 1, /*keepdim=* /true);  // Reduce axis 1 (columns), keep dims
 * std::cout << Z << std::endl;
 * // Z = [[2],
 * //      [2]]              // Indexes of max values per row
 * ```
 */
struct Argmax {   
    int axis; 
    bool keepdim;   

    constexpr type reduce(type dtype) const {
        return int64;
    } 

    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1);
        Shape out;
        for (size_t dim = 0; dim < shape.rank(); ++dim) {
            if (dim != static_cast<size_t>(axis)) out.expand(shape[dim]);
            else if (keepdim) out.expand(1);
        }
        return out;
    }
  
    void forward(Tensor const& input, Tensor& output) const;
}; 

/**
 * @brief Finds the **indexes of minimum values** along an axis.
 *
 * Identical to `Argmax` but for minima. Output dtype is `int64`.
 * Use `keepdim=true` to maintain shape for broadcasting.
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{3, 1, 4}, 
 *             {1, 5, 9}};
 *
 * Tensor Y = argmin(X, 1);
 * // Y = [1, 0]  // Min indexes per row
 * ```
 */
struct Argmin {   
    int axis;    
    bool keepdim;  
 
    constexpr type reduce(type dtype) const {
        return int64;
    }
 
    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1); 
        Shape reduced; 
        for (uint8_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension != static_cast<uint8_t>(axis))
                reduced.expand(shape[dimension]);
        }
        return reduced;
    } 
 
    void forward(Tensor const&, Tensor&) const; 
};

 
/**
 * @brief Sums tensor values along an axis.
 *
 * Preserves input dtype. Use `keepdim=true` to maintain shape for broadcasting.
 *
 * #### Example (NumPy/PyTorch behavior):
 * ```cpp
 * Tensor X = {{1, 2}, 
 *             {3, 4}};  // shape(2, 2)
 *
 * Tensor Y = sum(X, 0);  // Sum over rows
 * // Y = [4, 6]          // 1+3=4, 2+4=6
 *
 * Tensor Z = sum(X, 1, /*keepdim=* /true);
 * // Z = [[3],           // 1+2=3
 * //      [7]]           // 3+4=7
 * ```
 */
struct Argsum {
    int axis;       
    bool keepdim;  
    
    constexpr type reduce(type dtype) const {
        return dtype;   
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
 * @brief Computes the **mean** along an axis.
 *
 * Requires floating-point input (`float32`/`float64`).  
 * Use `keepdim=true` to maintain shape for broadcasting.
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{1.0, 2.0}, 
 *             {3.0, 4.0}};
 *
 * Tensor Y = mean(X, 0);
 * // Y = [2.0, 3.0]  // (1+3)/2=2.0, (2+4)/2=3.0
 * ```
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
 * @brief Creates an Argmax reduction.
 * @param axis Axis to reduce (`-1` for last axis). 
 * @param keepdim If `true`, keeps reduced axis as size 1. 
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{3, 1, 4}, 
 *             {1, 5, 9}};  // shape(2, 3)
 *
 * Tensor Y = argmax(X, 0);  // Reduce axis 0 (rows)
 * std::cout << Y << std::endl;
 * // Y = [0, 1, 1]          // Indexes of max values per column
 *
 * Tensor Z = argmax(X, 1, /*keepdim=* /true);  // Reduce axis 1 (columns), keep dims
 * std::cout << Z << std::endl;
 * // Z = [[2],
 * //      [2]]              // Indexes of max values per row
 * ```
 */
template<Expression Source>
constexpr auto argmax(Source&& source, int axis = -1, bool keepdim = false) {  
    return Reduction<Argmax, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim}, std::forward<Source>(source) 
    };
}
 

/**
 * @brief Creates an Argmin reduction.
 * @param axis Axis to reduce (`-1` for last axis). 
 * @param keepdim If `true`, keeps reduced axis as size 1. 
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{3, 1, 4}, 
 *             {1, 5, 9}};
 *
 * Tensor Y = argmin(X, 1);
 * // Y = [1, 0]  // Min indexes per row
 * ```
 */
template<Expression Source>
constexpr auto argmin(Source&& source, int axis = -1, bool keepdim = false) {  
    return Reduction<Argmin, Source>{ 
        {indexing::normalize(axis, source.shape().rank()), keepdim}, std::forward<Source>(source) 
    };
} 

/**
 * @brief Creates a sum reduction.
 * @param axis Axis to reduce (`-1` for last axis).
 * @param keepdim If `true`, keeps reduced axis as size 1.
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{1, 2, 3},
 *             {4, 5, 6}};  // shape(2, 3)
 *
 * Tensor Y = sum(X, 0);  // Reduce axis 0 (rows)
 * std::cout << Y << std::endl;
 * // Y = [5, 7, 9]       // Sum of values per column
 *
 * Tensor Z = sum(X, 1, /*keepdim=* /true);  // Reduce axis 1 (columns), keep dims
 * std::cout << Z << std::endl;
 * // Z = [[6],
 * //      [15]]           // Sum of values per row
 * ```
 */
template<Expression Source>
constexpr auto sum(Source&& source, int axis = -1, bool keepdim = false) {
    return Reduction<Argsum, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim},
        std::forward<Source>(source)
    };
}

/**
 * @brief Creates a mean reduction.
 * @param axis Axis to reduce (`-1` for last axis).
 * @param keepdim If `true`, keeps reduced axis as size 1.
 *
 * #### Example:
 * ```cpp
 * Tensor X = {{1.0, 2.0, 3.0},
 *             {4.0, 5.0, 6.0}};  // shape(2, 3)
 *
 * Tensor Y = mean(X, 0);  // Reduce axis 0 (rows)
 * std::cout << Y << std::endl;
 * // Y = [2.5, 3.5, 4.5]  // Mean of values per column
 *
 * Tensor Z = mean(X, 1, /*keepdim=* /true);  // Reduce axis 1 (columns), keep dims
 * std::cout << Z << std::endl;
 * // Z = [[2.0],
 * //      [5.0]]          // Mean of values per row
 * ```
 */
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