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

#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

/**
 * @file Operations.hpp
 * @author Eric Cardozo
 * @date 2025
 * @brief Defines expression templates tensor aritmetic operations.
 *
 * This header defines expression templates used to represent lazy evaluation of unary
 * and binary tensor aritmetic operations. The system enables composable arithmetic on tensor-like
 * objects while deferring actual computation until evaluation (via `forward()`).
 *
 * Includes:
 *  - Type promotion utility (`promote`)
 *  - Shape broadcasting logic (`broadcast`)
 *  - Unary and binary operation templates (`Unary`, `Binary`)
 *  - Arithmetic operation structs (`Addition`, `Subtraction`, etc.)
 *  - Overloaded operators (`+`, `-`, `*`)
 *
 * Part of the Tannic Tensor Library.
 */

#include <vector>  
#include <utility>      
#include <type_traits> 
 
#include "Concepts.hpp"
#include "Types.hpp"
#include "Shape.hpp"
#include "Strides.hpp"   
#include "Traits.hpp"
 
namespace tannic {

class Tensor;    
  
namespace operation {  

/**
 * @brief Expression template for a unary tensor aritmetic operation.
 *
 * Represents a lazily evaluated unary operation applied to a single tensor-like operand.
 * This template enables deferred execution and composability at compile time.
 *
 * @tparam Operation A stateless operation implementing `void forward(Tensor const&, Tensor&)`.
 * @tparam Operand An expression type satisfying the `Expression` concept.
 */
template<class Operation, Expression Operand>
class Unary { 
public:
    Operation operation;
    Trait<Operand>::Reference operand; 

    /**
     * @brief Constructs a Unary expression.
     * @param operation Operation functor (e.g., `Negation`).
     * @param operand Operand expression.
     */
    constexpr Unary(Operation operation, Trait<Operand>::Reference operand)
    :   operation(operation)
    ,   operand(operand)
    {}  
 
    /**
     * @brief Returns the data type of the result.
     *
     * Since the unary operation acts element-wise and does not change type,
     * this returns the same type as the operand.
     *
     * @return Data type of the operand.
     */
    constexpr type dtype() const {
        return operand.dtype();
    } 

    /**
     * @brief Returns the shape of the result.
     *
     * The output of a unary operation has the exact same shape as the operand.
     *
     * @return Shape of the operand.
     */
    constexpr Shape const& shape() const {
        return operand.shape();
    } 
 

    /**
     * @brief Returns the strides of the result.
     *
     * The output tensor has the same memory layout as the operand,
     * so the strides are forwarded directly.
     *
     * @return Strides of the operand.
     */
    constexpr Strides const& strides() const {
        return operand.strides();
    }  


    /**
     * @brief Returns the offset of the expression.
     *
     * Unary expressions are assumed to have zero offset, since the result is 
     * a new tensor.
     *
     * @return Always returns 0.
     */
    std::ptrdiff_t offset() const {
        return 0;
    }

    /**
     * @brief Evaluates the unary expression and returns a Tensor.
     * @return Resulting Tensor after applying the unary operation.
     */
    Tensor forward() const;
};

/**
 * @brief Expression template for a binary tensor operation.
 *
 * Represents a lazily evaluated binary operation between two expressions.
 * The `Binary` class performs type promotion, shape broadcasting, and stride generation
 * upon construction to prepare for efficient evaluation.
 *
 * @tparam Operation A stateless functor implementing `void forward(Tensor const&, Tensor const&, Tensor&)`.
 * @tparam Operand Left-hand-side expression satisfying the `Expression` concept.
 * @tparam Cooperand Right-hand-side expression satisfying the `Expression` concept.
 */
template<Operator Operation, Expression Operand, Expression Cooperand>
class Binary {
public:
    Operation operation;
    Trait<Operand>::Reference operand;
    Trait<Cooperand>::Reference cooperand;

    /**
     * @brief Constructs a Binary expression.
     *
     * - Computes `dtype` by promoting the operand and cooperand types.
     * - Computes `shape` via NumPy-style broadcasting.
     * - Computes `strides` based on the resulting shape.
     *
     * @param operation Binary operation functor (e.g., `Addition`).
     * @param operand Left-hand-side expression.
     * @param cooperand Right-hand-side expression.
     */
    constexpr Binary(Operation operation, Trait<Operand>::Reference operand, Trait<Cooperand>::Reference cooperand)
    :   operation(operation)
    ,   operand(operand)
    ,   cooperand(cooperand)
    ,   dtype_(operation.promote(operand.dtype(), cooperand.dtype()))
    ,   shape_(operation.broadcast(operand.shape(), cooperand.shape()))
    ,   strides_(shape_)
    {} 


    /**
     * @brief Returns the promoted data type of the result.
     *
     * The data type is computed using a type promotion rule (based on the `type` enum),
     * choosing the higher-precision or broader type between the two operands.
     *
     * @return Promoted type of the two operands.
     */
    constexpr type dtype() const {
        return dtype_;
    } 

    /**
     * @brief Returns the shape of the result.
     *
     * The shape is computed via broadcasting rules, similar to NumPy or PyTorch.
     * Dimensions are aligned from the right, and incompatible sizes throw.
     *
     * @return Broadcasted shape between the two operands.
     */
    constexpr Shape const& shape() const {
        return shape_;
    }

    /**
     * @brief Returns the output strides for the result tensor.
     *
     * The strides are computed based on the broadcasted output shape
     * and follow a standard row-major memory layout (like C arrays).
     *
     * @return Computed strides for the broadcasted shape.
     */
    constexpr Strides const& strides() const {
        return strides_;
    } 

    /**
     * @brief Returns the offset of the expression.
     *
     * Binary expressions are assumed to have zero offset. Since the result is 
     * a new tensor.
     *
     * @return Always returns 0.
     */
    std::ptrdiff_t offset() const {
        return 0;
    }

    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};       
  
    
/**
 * @brief Promotes two data types to the higher precision type.
 *
 * Given two tensor `type` values, this function returns the one
 * with greater numerical precedence based on their underlying enum values.
 * This is used to determine the result type in binary expressions.
 *
 * @param first The first data type.
 * @param second The second data type.
 * @return The promoted data type (the one with higher precedence). 
 */
static constexpr inline type promote(type first, type second) {
    return static_cast<int>(first) > static_cast<int>(second) ?  first : second;
} 

/**
 * @brief Computes the broadcasted shape of two tensors.
 *
 * This function implements NumPy-style broadcasting rules:
 * - Shapes are aligned from the trailing (rightmost) dimensions.
 * - Dimensions must match exactly or be 1; otherwise, an exception is thrown.
 * - The resulting shape uses the maximum dimension size at each axis.
 *
 * @param first The shape of the first tensor.
 * @param second The shape of the second tensor.
 * @return A new `Shape` that both inputs can be broadcast to.
 *
 * @throws std::invalid_argument if shapes are not broadcast-compatible.
 *
 * #### Example:
 * ```cpp
 * broadcast(Shape(3, 1), Shape(1, 4)) → Shape{3, 4}
 * broadcast(Shape(5)   , Shape(2, 5)) → Shape{2, 5}
 * broadcast(Shape(2, 3), Shape(3))    → Shape{2, 3}
 * ```
 */
static constexpr Shape broadcast(Shape const& first, Shape const& second) {
    auto first_rank = first.rank();
    auto second_rank = second.rank();
    auto rank = std::max(first_rank, second_rank);
    std::vector<std::size_t> result(rank, 1);

    for (auto dimension = 0; dimension < rank; ++dimension) {
        auto first_dimension = (dimension < rank - first_rank) ? 1 : first[dimension - (rank - first_rank)];
        auto second_dimension = (dimension < rank - second_rank) ? 1 : second[dimension - (rank - second_rank)];

    if (!(first_dimension == second_dimension || first_dimension == 1 || second_dimension == 1)) {
        throw std::invalid_argument("Shapes are not broadcast-compatible.");
    }
        result[dimension] = std::max(first_dimension, second_dimension);
    }
    return Shape(result);
} 


/**
 * @brief Unary element-wise negation of a tensor expression.
 *
 * Used to build a lazily evaluated `Unary<Negation, Operand>` expression.
 * The computation is not performed immediately, but deferred until 
 * `forward()` is called or the expression is assigned to a `Tensor` object.
 * 
 * #### Example: Unary negation
 * 
 * ```cpp
 * Tensor A(float32, {2, 2}); A.initialize();
 * A[0, 0] = 1; A[0, 1] = -2;
 * A[1, 0] = 3; A[1, 1] = -4;
 *
 * auto B = -A;            // Lazy negation (Unary<Negation, Tensor>)  
 * Tensor C = B.forward(); // Triggers computation
 *
 * Tensor D = -A;          // Computation triggered when assigned to Tensor
 *
 * std::cout << D;
 * // Output: [[-1, 2], [-3, 4]]
 * ```
 */
struct Negation {
    void forward(Tensor const&, Tensor&) const;   
};


/**
 * @brief Binary element-wise addition of two tensor expressions.
 *
 * Used to build a lazily evaluated `Binary<Addition, Augend, Addend>` expression.
 * Computation is deferred until `forward()` is called or assigned to a `Tensor`.
 * Supports broadcasting and type promotion.
 * 
 * #### Example: Addition
 * 
 * ```cpp
 * Tensor A(float32, {2, 2}); A.initialize();
 * A[0, 0] = 1; A[0, 1] = 2;
 * A[1, 0] = 3; A[1, 1] = 4;
 *
 * Tensor B(float32, {2, 2}); B.initialize();
 * B[0, 0] = 5; B[0, 1] = 6;
 * B[1, 0] = 7; B[1, 1] = 8;
 *
 * auto C = A + B;
 * std::cout << C.forward();
 * // Output: [[6, 8], [10, 12]]
 * ```
 */
struct Addition {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 

    constexpr static type promote(type first, type second) {
        return operation::promote(first, second);
    }

    constexpr static Shape broadcast(Shape const& first, Shape const& second) {
        return operation::broadcast(first, second);
    }
};


/**
 * @brief Binary element-wise multiplication of two tensor expressions.
 *
 * Used to build a lazily evaluated `Binary<Multiplication, Multiplicand, Multiplier>` expression.
 * Computation is deferred until `forward()` is called or assigned to a `Tensor`.
 * Supports broadcasting and type promotion.
 * 
 * #### Example: Multiplication
 * 
 * ```cpp
 * Tensor A(float32, {2, 2}); A.initialize();
 * A[0, 0] = 1; A[0, 1] = 2;
 * A[1, 0] = 3; A[1, 1] = 4;
 *
 * Tensor B(float32, {2, 2}); B.initialize();
 * B[0, 0] = 10; B[0, 1] = 20;
 * B[1, 0] = 30; B[1, 1] = 40;
 *
 * auto C = A * B;
 * std::cout << C.forward();
 * // Output: [[10, 40], [90, 160]]
 * ```
 */ 
struct Multiplication {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 

    constexpr static type promote(type first, type second) {
        return operation::promote(first, second);
    }

    constexpr static Shape broadcast(Shape const& first, Shape const& second) {
        return operation::broadcast(first, second);
    }
};


/**
 * @brief Binary element-wise subtraction of two tensor expressions.
 *
 * Used to build a lazily evaluated `Binary<Subtraction, Subtrahend, Minuend>` expression.
 * Computation is deferred until `forward()` is called or assigned to a `Tensor`.
 * Supports broadcasting and type promotion.
 * 
 * #### Example: Subtraction
 * 
 * ```cpp
 * Tensor A(float32, {2, 2}); A.initialize();
 * A[0, 0] = 5; A[0, 1] = 7;
 * A[1, 0] = 9; A[1, 1] = 11;
 *
 * Tensor B(float32, {2, 2}); B.initialize();
 * B[0, 0] = 1; B[0, 1] = 2;
 * B[1, 0] = 3; B[1, 1] = 4;
 *
 * auto C = A - B;
 * std::cout << C.forward();
 * // Output: [[4, 5], [6, 7]]
 * ```
 */
struct Subtraction {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 

    constexpr static type promote(type first, type second) {
        return operation::promote(first, second);
    }

    constexpr static Shape broadcast(Shape const& first, Shape const& second) {
        return operation::broadcast(first, second);
    }
}; 


/**
 * @brief Element-wise negation of a tensor expression.
 *
 * Constructs a lazily evaluated `Unary<Negation, Operand>` expression.
 * This operation does not perform immediate computation; it represents a node
 * in the expression graph and will be evaluated upon calling `forward()`.
 *
 * @tparam Operand A type satisfying the `Expression` concept.
 * @param operand The input expression to negate.
 * @return A unary expression representing element-wise negation.
 */
template<Expression Operand>
constexpr auto operator-(Operand&& operand) {
    return Unary<Negation, Operand>{{}, std::forward<Operand>(operand)};
} 


/**
 * @brief Element-wise addition of two tensor expressions.
 *
 * Constructs a lazily evaluated `Binary<Addition, Augend, Addend>` expression.
 * The result's shape is determined via broadcasting rules, and the resulting
 * data type is the promoted type of the two operands. 
 * The computation on the elements is not performed immediately, but is 
 * deferred until forward() is called or the expression is assigned to 
 * a Tensor object.
 *
 * @tparam Augend Left-hand-side operand (expression).
 * @tparam Addend Right-hand-side operand (expression).
 * @param augend First operand.
 * @param addend Second operand.
 * @return A binary expression representing element-wise addition.
 */
template<Expression Augend, Expression Addend>
constexpr auto operator+(Augend&& augend, Addend&& addend) {
    return Binary<Addition, Augend, Addend>{{}, std::forward<Augend>(augend), std::forward<Addend>(addend)};
}


/**
 * @brief Element-wise subtraction of two tensor expressions.
 *
 * Constructs a lazily evaluated `Binary<Subtraction, Subtrahend, Minuend>` expression.
 * Shapes are broadcasted from the operands, and the type is promoted accordingly.
 * The computation on the elements is not performed immediately, but is 
 * deferred until forward() is called or the expression is assigned to 
 * a Tensor object.
 *
 * @tparam Subtrahend Left-hand-side operand (expression).
 * @tparam Minuend Right-hand-side operand (expression).
 * @param subtrahend First operand.
 * @param minuend Second operand.
 * @return A binary expression representing element-wise subtraction.
 */
template<Expression Subtrahend, Expression Minuend>
constexpr auto operator-(Subtrahend&& subtrahend, Minuend&& minuend) {
    return Binary<Subtraction, Subtrahend, Minuend>{{}, std::forward<Subtrahend>(subtrahend), std::forward<Minuend>(minuend)};
}


/**
 * @brief Element-wise multiplication of two tensor expressions.
 *
 * Constructs a lazily evaluated `Binary<Multiplication, Multiplicand, Multiplier>` expression.
 * As with other binary operations, this uses broadcasting and type promotion internally.
 * The computation on the elements is not performed immediately, but is 
 * deferred until forward() is called or the expression is assigned to 
 * a Tensor object.
 * 
 * @tparam Multiplicand Left-hand-side operand (expression).
 * @tparam Multiplier Right-hand-side operand (expression).
 * @param multiplicand First operand.
 * @param multiplier Second operand.
 * @return A binary expression representing element-wise multiplication.
 */
template<Expression Multiplicand, Expression Multiplier>
constexpr auto operator*(Multiplicand&& multiplicand, Multiplier&& multiplier) {
    return Binary<Multiplication, Multiplicand, Multiplier>{{}, std::forward<Multiplicand>(multiplicand), std::forward<Multiplier>(multiplier)};
} 

} // namespace operation


using operation::operator-; 
using operation::operator+; 
using operation::operator*;  


} // namespace tannic

#endif // OPERATIONS_HPP 