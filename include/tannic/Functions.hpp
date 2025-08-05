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

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

/**
 * @file Functions.hpp
 * @author Eric Cardozo
 * @date 2025
 * @brief Defines mathematical function operations for tensor expressions.
 *
 * This header provides lazy-evaluated mathematical functions for tensor-like objects,
 * implemented as expression templates. All operations are element-wise and maintain
 * the original tensor's shape and strides.
 *
 * Includes the following mathematical functions:
 * - Basic functions:
 *   * log() - Natural logarithm (ln)
 *   * exp() - Exponential (e^x)
 *   * sqrt() - Square root (√x)
 *   * abs() - Absolute value (|x|)
 * - Trigonometric functions (radians):
 *   * sin() - Sine
 *   * cos() - Cosine
 *   * tan() - Tangent
 * - Hyperbolic functions:
 *   * sinh() - Hyperbolic sine
 *   * cosh() - Hyperbolic cosine
 *   * tanh() - Hyperbolic tangent
 * 
 * Part of the Tannic Tensor Library.
 */

#include "Concepts.hpp"
#include "Traits.hpp" 
#include "Shape.hpp"
#include "Strides.hpp"
#include "Tensor.hpp" 

namespace tannic {
  
namespace function {

/**
 * @brief Expression template for mathematical function operations.
 *
 * Represents a lazily evaluated unary functor that takes a function as argument
 * and maps it to an argument.
 * The actual computation is deferred until `forward()` is called.
 *
 * @tparam Function A mathematical function satisfying the Functional concept`
 * @tparam Argument An expression type satisfying the `Expression` concept
 */
template<Functional Function, Expression Argument>
class Functor {
public:
    Function function;
    typename Trait<Argument>::Reference argument;

    /**
     * @brief Constructs a Function expression
     * @param function The function (e.g., Log, Exp)
     * @param argument The input tensor expression
     */
    constexpr Functor(Function function, typename Trait<Argument>::Reference argument)
    :   function(function)
    ,   argument(argument)
    {}

    /**
     * @brief Returns the data type of the result.
     * @return The data type of the underlying argument.
     *
     * Functions in this expression template system are type-preserving —
     * applying them element-wise does not change the scalar type
     * (e.g., applying `exp()` to a `float32` tensor produces `float32` output).
     * Therefore, we can directly query and return the argument's dtype.
     */
    constexpr type dtype() const {
        return argument.dtype();
    }


    /**
     * @brief Returns the shape of the result.
     * @return A const reference to the shape of the underlying argument.
     *
     * The shape is returned as `const&` to avoid copying the shape object,
     * and because element-wise functions do not alter tensor dimensions.
     */
    constexpr Shape const& shape() const {
        return argument.shape();
    }  

    /**
     * @brief Returns the strides of the result.
     * @return A const reference to the strides of the underlying argument.
     *
     * Strides describe memory layout, and for pure element-wise operations
     * they remain identical to those of the input tensor.
     * Returning `const&` avoids copying and preserves the original stride information.
     */
    constexpr Strides const& strides() const {
        return argument.strides();
    }

    /**
     * @brief Returns the offset of the result.
     * @return The offset (in elements) into the underlying tensor storage.
     *
     * Since element-wise functions do not change the memory location of the data,
     * the offset is taken directly from the argument.
     */
    auto offset() const {
        return argument.offset();
    }
 
    Tensor forward() const {
        Tensor source = argument.forward();
        Tensor target(dtype(), shape(), strides(), offset());
        function(source, target);
        return target;
    } 
};

/**
 * @brief Functor natural logarithm (ln(x))
 * Applies element-wise natural logarithm to tensor elements
 */
struct Log {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor exponential (e^x)
 * Applies element-wise exponential to tensor elements
 */
struct Exp {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor square root (√x)
 * Applies element-wise square root to tensor elements
 */
struct Sqrt { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor absolute value (|x|)
 * Applies element-wise absolute value to tensor elements
 */
struct Abs { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor sine
 * Applies element-wise sine to tensor elements (radians)
 */
struct Sin {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor cosine
 * Applies element-wise cosine to tensor elements (radians)
 */
struct Cos { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor tangent
 * Applies element-wise tangent to tensor elements (radians)
 */
struct Tan { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor hyperbolic sine
 * Applies element-wise hyperbolic sine to tensor elements
 */
struct Sinh { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor hyperbolic cosine
 * Applies element-wise hyperbolic cosine to tensor elements
 */
struct Cosh { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Functor hyperbolic tangent
 * Applies element-wise hyperbolic tangent to tensor elements
 */
struct Tanh { 
    void operator()(Tensor const&, Tensor&) const;
};
 
/**
 * @brief Creates a lazy-evaluated natural logarithm expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise ln(operand)
 * @note Computes natural logarithm (base e) for each element
 * @see Log
 */
template<Expression Operand>
constexpr auto log(Operand&& operand) {
    return Functor<Log, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated exponential function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise e^(operand)
 * @see Exp
 */
template<Expression Operand>
constexpr auto exp(Operand&& operand) {
    return Functor<Exp, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated square root expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise √operand
 * @note Returns NaN for negative inputs
 * @see Sqrt
 */
template<Expression Operand>
constexpr auto sqrt(Operand&& operand) {
    return Functor<Sqrt, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated absolute value expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise |operand|
 * @see Abs
 */
template<Expression Operand>
constexpr auto abs(Operand&& operand) {
    return Functor<Abs, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated sine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise sin(operand)
 * @see Sin
 */
template<Expression Operand>
constexpr auto sin(Operand&& operand) {
    return Functor<Sin, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated cosine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise cos(operand)
 * @see Cos
 */
template<Expression Operand>
constexpr auto cos(Operand&& operand) {
    return Functor<Cos, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated tangent function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise tan(operand)
 * @note Returns NaN where cosine equals zero
 * @see Tan
 */
template<Expression Operand>
constexpr auto tan(Operand&& operand) {
    return Functor<Tan, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic sine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise sinh(operand)
 * @see Sinh
 */
template<Expression Operand>
constexpr auto sinh(Operand&& operand) {
    return Functor<Sinh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic cosine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise cosh(operand)
 * @see Cosh
 */
template<Expression Operand>
constexpr auto cosh(Operand&& operand) {
    return Functor<Cosh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic tangent expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise tanh(operand) 
 * @see Tanh
 */
template<Expression Operand>
constexpr auto tanh(Operand&& operand) {
    return Functor<Tanh, Operand>({}, std::forward<Operand>(operand));
}

} // namespace function

using function::log;
using function::exp;
using function::sqrt;
using function::abs;
using function::sin;
using function::cos;
using function::tan;
using function::sinh;
using function::cosh;
using function::tanh;

} //namespace tannic

#endif // FUNCTIONS_HPP