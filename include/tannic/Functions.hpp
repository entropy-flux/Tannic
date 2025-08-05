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
 * Represents a lazily evaluated unary function applied to a tensor expression.
 * The actual computation is deferred until `forward()` is called.
 *
 * @tparam Functor A stateless functor implementing `operator()(Tensor const&, Tensor&)`
 * @tparam Operand An expression type satisfying the `Expression` concept
 */
template<class Functor, Expression Operand>
class Function {
public:
    Functor functor;
    typename Trait<Operand>::Reference operand;

    /**
     * @brief Constructs a Function expression
     * @param functor The function functor (e.g., Log, Exp)
     * @param operand The input tensor expression
     */
    constexpr Function(Functor functor, typename Trait<Operand>::Reference operand)
    :   functor(functor)
    ,   operand(operand)
    {}

    /**
     * @brief Returns the data type of the result
     * @return Data type of the operand (functions preserve input type)
     */
    constexpr type dtype() const {
        return operand.dtype();
    }

    /**
     * @brief Returns the shape of the result
     * @return Shape of the operand (functions preserve input shape)
     */
    constexpr Shape const& shape() const {
        return operand.shape();
    }  

    /**
     * @brief Returns the strides of the result
     * @return Strides of the operand (functions preserve input strides)
     */
    constexpr Strides const& strides() const {
        return operand.strides();
    }

    /**
     * @brief Returns the offset of the operand
     * @return Offset of the underlying tensor data
     */
    auto offset() const {
        return operand.offset();
    }

    /**
     * @brief Evaluates the function expression
     * @return New Tensor with the function applied element-wise
     */
    Tensor forward() const {
        Tensor source = operand.forward();
        Tensor target(dtype(), shape());
        functor(source, target);
        return target;
    } 
};

/**
 * @brief Function natural logarithm (ln(x))
 * Applies element-wise natural logarithm to tensor elements
 */
struct Log {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function exponential (e^x)
 * Applies element-wise exponential to tensor elements
 */
struct Exp {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function square root (√x)
 * Applies element-wise square root to tensor elements
 */
struct Sqrt { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function absolute value (|x|)
 * Applies element-wise absolute value to tensor elements
 */
struct Abs { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function sine
 * Applies element-wise sine to tensor elements (radians)
 */
struct Sin {
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function cosine
 * Applies element-wise cosine to tensor elements (radians)
 */
struct Cos { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function tangent
 * Applies element-wise tangent to tensor elements (radians)
 */
struct Tan { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function hyperbolic sine
 * Applies element-wise hyperbolic sine to tensor elements
 */
struct Sinh { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function hyperbolic cosine
 * Applies element-wise hyperbolic cosine to tensor elements
 */
struct Cosh { 
    void operator()(Tensor const&, Tensor&) const;
};

/**
 * @brief Function hyperbolic tangent
 * Applies element-wise hyperbolic tangent to tensor elements
 */
struct Tanh { 
    void operator()(Tensor const&, Tensor&) const;
};
 
/**
 * @brief Creates a lazy-evaluated natural logarithm expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise ln(operand)
 * @note Computes natural logarithm (base e) for each element
 * @see Log
 */
template<Expression Operand>
constexpr auto log(Operand&& operand) {
    return Function<Log, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated exponential function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise e^(operand)
 * @see Exp
 */
template<Expression Operand>
constexpr auto exp(Operand&& operand) {
    return Function<Exp, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated square root expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise √operand
 * @note Returns NaN for negative inputs
 * @see Sqrt
 */
template<Expression Operand>
constexpr auto sqrt(Operand&& operand) {
    return Function<Sqrt, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated absolute value expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise |operand|
 * @see Abs
 */
template<Expression Operand>
constexpr auto abs(Operand&& operand) {
    return Function<Abs, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated sine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Function expression representing element-wise sin(operand)
 * @see Sin
 */
template<Expression Operand>
constexpr auto sin(Operand&& operand) {
    return Function<Sin, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated cosine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Function expression representing element-wise cos(operand)
 * @see Cos
 */
template<Expression Operand>
constexpr auto cos(Operand&& operand) {
    return Function<Cos, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated tangent function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Function expression representing element-wise tan(operand)
 * @note Returns NaN where cosine equals zero
 * @see Tan
 */
template<Expression Operand>
constexpr auto tan(Operand&& operand) {
    return Function<Tan, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic sine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise sinh(operand)
 * @see Sinh
 */
template<Expression Operand>
constexpr auto sinh(Operand&& operand) {
    return Function<Sinh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic cosine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise cosh(operand)
 * @see Cosh
 */
template<Expression Operand>
constexpr auto cosh(Operand&& operand) {
    return Function<Cosh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic tangent expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Function expression representing element-wise tanh(operand) 
 * @see Tanh
 */
template<Expression Operand>
constexpr auto tanh(Operand&& operand) {
    return Function<Tanh, Operand>({}, std::forward<Operand>(operand));
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