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

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP 

/**
 * @file transformations.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Defines tensor transformation operations.
 *
 * This header provides tensor transformation operations implemented as expression templates. 
 *
 * Part of the Tannic Tensor Library.
 */ 
#include <tuple>
#include <array>
#include <vector>
#include <cassert>

#include "concepts.hpp"
#include "types.hpp"
#include "traits.hpp"
#include "shape.hpp" 
#include "tensor.hpp" 
#include "exceptions.hpp"

namespace tannic {

class Tensor;  

} namespace tannic::expression {

/**
 * @brief Expression template for tensor transformations
 *
 * Represents a lazily evaluated transformation operation between multiple tensors.
 * Handles type promotion, shape broadcasting, and proper stride computation.
 *
 * @tparam Operation The transformation operation type
 * @tparam Operands Variadic template for input expressions
 */
template<class Operation, Composable ... Operands>
class Transformation : public Expression<Operation, Operands...> {
public:  
    constexpr Transformation(Operation operation, typename Trait<Operands>::Reference... operands)  
    :   Expression<Operation, Operands...>(operation, operands...)
    ,   dtype_(operation.promote(operands.dtype()...))
    ,   shape_(operation.transform(operands.shape()...)) 
    ,   strides_(shape_)
    {}
     
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
        Tensor result(dtype_, shape_);
        std::apply([&](const auto&... arguments) {
            return this->operation.forward(arguments.forward()..., result);
        }, this->operands);
        return result;
    }

private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
};

} namespace tannic::transformation {

using tannic::expression::Transformation;

/**
 * @brief Helper function to compute type promotion table indices
 * @param first  First type
 * @param second Second type
 * @return Index into promotion table
 */
static constexpr auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(second) * static_cast<int>(TYPES);
}  

/**
 * @brief Transformation composition (Known as Matrix Multiplication) operation
 *
 * Implements tensor composition with:
 * - Automatic type promotion
 * - Shape broadcasting for batch dimensions
 * - Support for vectors, matrices, and higher-rank tensors
 */ 
struct Composition {    
    double scale = 1.0;

    /**
     * @brief Type promotion rules table
     *
     * Defines promotion rules for all type combinations:
     * - Integer operations promote to avoid overflow
     * - Mixed integer/float promotes to float
     * - Operations preserve highest precision
     */
    static constexpr auto promotions = []() {
        std::array<type, index(TYPES, TYPES)> table{};  
        table.fill(unknown);
        // Integer promotions
        table[index(int8, int8)]   = int32;
        table[index(int8, int16)]  = int32;
        table[index(int8, int32)]  = int32;
        table[index(int8, int64)]  = int64;  

        table[index(int16, int8)]    = int32;
        table[index(int16, int16)]   = int32;
        table[index(int16, int32)]   = int32;
        table[index(int16, int64)]   = int64; 
        
        table[index(int32, int8)]    = int32;
        table[index(int32, int16)]   = int32;
        table[index(int32, int32)]   = int64;
        table[index(int32, int64)]   = int64; 
        
        table[index(int64, int8)]    = int64;
        table[index(int64, int16)]   = int64;
        table[index(int64, int32)]   = int64;
        table[index(int64, int64)]   = int64;  
    
        table[index(int8, float16)]   = float16;
        table[index(float16, int8)]   = float16;
        table[index(int16, float16)]  = float16;
        table[index(float16, int16)]  = float16;
        table[index(int32, float16)]  = float32;  
        table[index(float16, int32)]  = float32;
        table[index(int64, float16)]  = float64;   
        table[index(float16, int64)]  = float64;  

        table[index(float16, float16)] = float16;
        table[index(float16, float32)] = float32;
        table[index(float32, float16)] = float32;
        table[index(float16, float64)] = float64;
        table[index(float64, float16)] = float64;
 
        table[index(int8, bfloat16)]   = bfloat16;
        table[index(bfloat16, int8)]   = bfloat16;
        table[index(int16, bfloat16)]  = bfloat16;
        table[index(bfloat16, int16)]  = bfloat16;
        table[index(int32, bfloat16)]  = float32;
        table[index(bfloat16, int32)]  = float32;
        table[index(int64, bfloat16)]  = float64;
        table[index(bfloat16, int64)]  = float64;

        table[index(bfloat16, bfloat16)] = bfloat16;
        table[index(bfloat16, float16)]  = float16;
        table[index(float16, bfloat16)]  = float16;
        table[index(bfloat16, float32)]  = float32;
        table[index(float32, bfloat16)]  = float32;
        table[index(bfloat16, float64)]  = float64;
        table[index(float64, bfloat16)]  = float64;

        table[index(int32, float32)] = float32;
        table[index(float32, int32)] = float32;
        table[index(int32, float64)] = float64;
        table[index(float64, int32)] = float64; 
         
        table[index(float32, float32)] = float32;
        table[index(float32, float64)] = float64;
        table[index(float64, float32)] = float64;
        table[index(float64, float64)] = float64; 
        return table;
    }();  
 
    /**
     * @brief Promotes two operand types to a common type for composition operations
     * @param inner Type of the inner (right) operand
     * @param outer Type of the outer (left) operand 
     * @return Promoted type according to the composition rules
     * @throws assertion error if the type combination is unsupported
     *
     * @details This promotion system:
     * 1. Uses a precomputed promotion table for all valid type combinations
     * 2. Ensures safe arithmetic by:
     *    - Promoting integers to avoid overflow (e.g., int8 → int32)
     *    - Promoting mixed integer/float to float
     *    - Preserving higher precision when types differ
     * 3. Explicitly rejects invalid combinations via assertion
     *
     * Example promotions:
     * 
     * ```
     * promote(int8, int32) → int32  // Integer widening
     * promote(int32, float32) → float32  // Mixed to float
     * promote(float32, float64) → float64  // Precision preservation
     * ```
     */
    constexpr static type promote(type inner, type outer) {
        type dtype = promotions[index(inner, outer)];
        if (dtype == unknown) 
            throw Exception("Unsuported dtypes");
        return dtype;
    }

    /**
     * @brief Computes transformed output shape for composition
     * @param first First operand shape
     * @param second Second operand shape
     * @return Broadcasted output shape
     * @throws assertion error if shapes are incompatible
     *
     * @details Handles multiple cases:
     * 1. Vector-vector: (n) × (n) → scalar (empty shape)
     * 2. Matrix-vector: (m,n) × (n) → (m)
     * 3. Vector-matrix: (n) × (n,k) → (k)
     * 4. Matrix-matrix: (m,n) × (n,k) → (m,k)
     * 5. Batched operations: (...,m,n) × (...,n,k) → (...,m,k)
     */
    static constexpr Shape transform(Shape const& first, Shape const& second) {
        auto first_rank = first.rank();
        auto second_rank = second.rank();
          
        if (first_rank == 1 && second_rank == 1) {
            if (first_rank != 1 | second_rank != 1)
                throw Exception("dimensions must match for dot product");
            return Shape{};  // Scalar result
        }
          
        if (first_rank == 1 && second_rank == 2) {
            if (first[0] != second[0])
                throw Exception("Matrix inner dimensions do not match");
            if (first[0] != second[0])
            return Shape{second[0]};  // Vector result
        }
         
        // Vector-matrix multiplication
        if (first_rank == 2 && second_rank == 1) {
            if (first[1] != second[0])
                throw Exception("Matrix inner dimensions do not match");
            return Shape{first[0]};  // Vector result
        }  

        if (first_rank < 2 | second_rank < 2) 
            throw Exception("Inputs must have rank >= 2");
        
        // Handle batch dimensions
        Shape first_batches(first.begin(), first.end() - 2);
        Shape second_batches(second.begin(), second.end() - 2);
        Shape batches = operation::broadcast(first_batches, second_batches);
        
        // Check matrix inner dimensions
        auto K1 = *(first.end() - 1);
        auto K2 = *(second.end() - 2);   
        assert(K1 == K2 && "Inner dimensions must match for matmul");
        
        // Get output matrix dimensions
        auto M = *(first.end() - 2);
        auto N = *(second.end() - 1);
        
        // Combine batch and matrix dimensions
        std::vector<Shape::size_type> result(batches.begin(), batches.end());
        result.push_back(M);
        result.push_back(N);
        return Shape(result);
    }

    void forward(Tensor const& outer, Tensor const& inner, Tensor& result) const; 
}; 
 
/**
 * @brief Represents the outer product operation between two vectors. 
 * 
 * The outer product of two vectors with shapes (n) and (m) results in 
 * a matrix of shape (n, m). Currently, only vectors are supported — tensors 
 * with rank greater than 1 are not supported, but this may be extended in the future.
 */
struct Outer { 
    /**
     * @brief Type promotion for the outer product operation
     * 
     * Promotes two operand types to the higher precision type.  
     * @param first Type of the first operand (left)
     * @param second Type of the second operand (right)
     * @return Promoted type for outer product computation
     */
    static constexpr type promote(type first, type second) {
        return static_cast<int>(first) > static_cast<int>(second) ? first : second;
    }

    /**
     * @brief Computes output shape for the outer product of two vectors 
     * @param first Shape of the first tensor 
     * @param second Shape of the second tensor  
     * @return Shape of the resulting outer product tensor
     * 
     * @throws assertion error if either input tensor is not rank 1
     * for now outer product only support vectors and not general tensors
     * but this may change in the future.
     * 
     * @details The outer product of two vectors with shapes (n) and (m) 
     * produces a matrix of shape (n, m).
     * Outer product is not defined here for tensors with rank > 1.
     */
    static constexpr Shape transform(Shape const& first, Shape const& second) {
        if(first.rank() != 1 | second.rank() != 1) 
            throw Exception("Outer product of tensors with rank more than 1 not supported");
        return Shape(first[0], second[0]);
    }
 
    void forward(Tensor const& first, Tensor const& second, Tensor& result) const;
};  

/**
 * @brief Repetition operation along a specified axis.
 * 
 * Replicates the tensor data along a given axis multiple times.
 */
struct Repetition {
    int repeats;
    int axis;
     
    /**
     * @brief Type promotion for repetition operation.
     * 
     * Repetition does not change data type.
     * 
     * @param dtype Original tensor data type.
     * @return Same data type.
     */
    constexpr type promote(type dtype) const {
        return dtype;
    }

    /**
     * @brief Computes output shape after repetition.
     * 
     * Multiplies the size of the specified axis by repeats.
     * 
     * @param shape Input tensor shape.
     * @return Output shape after repetition.
     */
    constexpr Shape transform(Shape shape) const {
        shape[indexing::normalize(axis, shape.rank())] *= repeats; 
        return shape;
    }

    void forward(Tensor const&, Tensor&) const;
};


/**
 * @brief Concatenation operation along a specified axis.
 * 
 * Concatenates two tensors of the same rank and dtype along a given axis.
 */
struct Concatenation {
    int axis;
    /**
     * @brief Type promotion for concatenation.
     * 
     * Requires both tensors to have identical data types.
     * 
     * @param first Left tensor dtype.
     * @param second Right tensor dtype.
     * @return Data type (same as inputs).
     */
    constexpr auto promote(type first, type second) const {
        if(first != second)
            throw Exception("Cannot concatenate tensors of different dtypes");
        return first;
    } 

    /**
     * @brief Computes output shape after concatenation.
     * 
     * Validates that all dimensions except the concatenation axis are equal.
     * Output shape is input shapes with concatenation axis dimension summed.
     * 
     * @param first Left tensor shape.
     * @param second Right tensor shape.
     * @return Output shape after concatenation.
     */
    constexpr Shape transform(Shape const& first, Shape const& second) {
        assert(first.rank() == second.rank() && "Ranks must match for concatenation");
        Shape result = first;

        for (int dimension = 0; dimension < first.rank(); ++dimension) {
            if (dimension == axis) {
                result[dimension] = first[dimension] + second[dimension];
            } else {
                if (first[dimension] != second[dimension])
                    throw Exception("All dimensions except concat axis must match");
            }
        }

        return result;
    }

    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

/**
 * @brief Repack operation (makes a tensor contiguous in memory)
 *
 * Ensures the tensor is stored contiguously (row-major order).
 * Does not change shape or dtype.
 */
struct Repack {
    /**
     * @brief Type promotion for repack.
     * 
     * Repack does not change dtype.
     */
    constexpr type promote(type dtype) const {
        return dtype;
    }

    /**
     * @brief Computes output shape after repack.
     *
     * Repack does not change the shape.
     */
    constexpr Shape transform(Shape const& shape) const {
        return shape;
    }

    /**
     * @brief Copies the tensor data to a contiguous layout if needed.
     * 
     * @param source Input tensor.
     * @param result Output tensor with contiguous layout.
     */
    void forward(Tensor const& source, Tensor& result) const;
}; 

/**
 * @brief Creates a composition (matrix multiplication) expression
 * @tparam Outer Outer tensor expression type
 * @tparam Inner Inner tensor expression type
 * @param outer Outer tensor operand
 * @param inner Inner tensor operand
 * @return Transformation expression representing the composition
 */
template<Composable Outer, Composable Inner>
constexpr auto composition(Outer&& outer, Inner&& inner, double scale) {
    return Transformation<Composition, Outer, Inner>{
        {scale}, 
        std::forward<Outer>(outer), 
        std::forward<Inner>(inner)
    };
} 

/**
 * @brief Creates an outer product expression.
 * @tparam First Left tensor expression type
 * @tparam Second Right tensor expression type
 * @param first Left tensor operand
 * @param second Right tensor operand
 * @return Transformation expression representing the outer product
 */
template<Composable First, Composable Second>
constexpr auto outer(First&& first, Second&& second) {
    return Transformation<Outer, First, Second>(
        {},
        std::forward<First>(first),
        std::forward<Second>(second)
    );
}   


/**
 * @brief Creates a repetition transformation.
 * 
 * @tparam Source Source tensor expression type.
 * @param source Tensor to be repeated.
 * @param repeats Number of repetitions.
 * @param axis Axis along which to repeat.
 * @return Transformation expression representing repetition.
 */
template<Composable Source>
constexpr auto repeat(Source&& source, int repeats, int axis = 0) {
    return Transformation<Repetition, Source>(
        {repeats, indexing::normalize(axis, source.shape().rank())},
        std::forward<Source>(source)
    ); 
}


/**
 * @brief Helper function to create a concatenation transformation.
 * 
 * @tparam First Left tensor expression type.
 * @tparam Second Right tensor expression type.
 * @param first Left tensor operand.
 * @param second Right tensor operand.
 * @param axis Axis along which to concatenate.
 * @return Transformation expression representing concatenation.
 */
template<Composable First, Composable Second>
constexpr auto concatenate(First&& first, Second&& second, int axis = 0) {
    if(axis < 0) 
        throw Exception("Negative index not supported in concat");
        
    return Transformation<Concatenation, First, Second>(
        {axis},
        std::forward<First>(first),
        std::forward<Second>(second)
    ); 
}

/**
 * @brief Creates a repack transformation. 
 * Ensures the tensor is stored contiguously (row-major order).
 * Does not change shape or dtype.
 *
 * @tparam Source Source tensor expression type.
 * @param source Tensor to be repacked.
 * @return Transformation expression representing repack.
 */
template<Composable Source>
constexpr auto repack(Source&& source) {
    return Transformation<Repack, Source>(
        {},
        std::forward<Source>(source)
    ); 
}
 
/**
 * @brief Creates a view but always repacks the tensor 
 * into a contiguous layout. 
 *
 * @tparam Source Source tensor expression type.
 * @param source Tensor to be reshaped.
 * @return Transformation expression representing reshape.
 */
template<Composable Source, Integral ... Indexes>
constexpr auto reshape(Source&& source, Indexes ... indexes) {
    return expression::View<Transformation<Repack, Source>>(
        repack(source), indexes...
    );
} 

} namespace tannic {

using transformation::outer;
using transformation::repeat;
using transformation::concatenate;
using transformation::repack;
using transformation::reshape; 

/**
 * @brief Matrix multiplication convenience function
 * @tparam Multiplicand Left tensor expression type
 * @tparam Multiplier Right tensor expression type
 * @param multiplicand Left tensor operand
 * @param multiplier Right tensor operand
 * @return Transformation expression representing matrix multiplication
 */
template<Composable Multiplicand, Composable Multiplier>
constexpr auto matmul(Multiplicand&& multiplicand, Multiplier&& multiplier, double scale = 1.0) {
    return transformation::composition(
        std::forward<Multiplicand>(multiplicand),
        std::forward<Multiplier>(multiplier),
        scale
    );
}

} // namespace tannic

#endif // TRANSFORMATIONS_HPP