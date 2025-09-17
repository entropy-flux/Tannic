// Copyright 2025 Eric Hermosis
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

// Copyright 2025 Eric Hermosis
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
 * @file reductions.hpp
 * @author Eric Hermosis
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
#include "concepts.hpp"
#include "expressions.hpp"
#include "types.hpp"
#include "traits.hpp"
#include "shape.hpp" 
#include "tensor.hpp" 
#include "indexing.hpp"
#include "exceptions.hpp"

namespace tannic::expression {
    
template<class Reducer, Composable Operand>
class Reduction : public Expression<Reducer, Operand> {
public: 
    constexpr Reduction(Reducer reducer, typename Trait<Operand>::Reference operand)
    :   Expression<Reducer, Operand>(reducer, operand)
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
 
    Tensor forward(Context const& context) const {   
        Tensor source = std::get<0>(this->operands).forward(context);
        Tensor result(this->dtype(),shape(), strides(), offset());          
        this->operation.forward(source, result);
        return result;
    } 

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
}; 

struct Argmax {   
    int axis; 
    bool keepdim;   

    constexpr type reduce(type dtype) const {
        return int64;
    } 

    constexpr Shape reduce(Shape const& shape) const {
        if (shape.rank() == 0) 
            throw Exception("Cannot reduce scalar tensors");

        Shape out;
        for (size_t dim = 0; dim < shape.rank(); ++dim) {
            if (dim != static_cast<size_t>(axis)) out.expand(shape[dim]);
            else if (keepdim) out.expand(1);
        }
        return out;
    }
  
    void forward(Tensor const& input, Tensor& output) const;
}; 


struct Argmin {   
    int axis;    
    bool keepdim;  
 
    constexpr type reduce(type dtype) const {
        return int64;
    }
 
    constexpr Shape reduce(Shape const& shape) const {
        if (shape.rank() == 0) 
            throw Exception("Cannot reduce scalar tensors");

        Shape reduced; 
        for (uint8_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension != static_cast<uint8_t>(axis))
                reduced.expand(shape[dimension]);
        }
        return reduced;
    } 
 
    void forward(Tensor const&, Tensor&) const; 
};

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

/********************************************************************************/
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
template<Composable Source>
constexpr auto argmax(Source&& source, int axis, bool keepdim = false) {  
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
template<Composable Source>
constexpr auto argmin(Source&& source, int axis, bool keepdim = false) {  
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
template<Composable Source>
constexpr auto sum(Source&& source, int axis, bool keepdim = false) {
    return Reduction<Argsum, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim},
        std::forward<Source>(source)
    };
}

/**
 * @brief Creates a mean reduction.
 * @param axis Axis to reduce.
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
template<Composable Source>
constexpr auto mean(Source&& source, int axis, bool keepdim = false) {
    return Reduction<Argmean, Source>{
        {indexing::normalize(axis, source.shape().rank()), keepdim},
        std::forward<Source>(source)
    };
}
 
} namespace tannic {
 
using expression::argmax;
using expression::argmin;
using expression::sum;
using expression::mean;

} // namespace tannic

#endif // REDUCTIONS_HPP