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

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
 
#include "concepts.hpp"
#include "expressions.hpp"
#include "traits.hpp" 
#include "shape.hpp"
#include "strides.hpp"
#include "tensor.hpp"  

namespace tannic::expression {  
    
template<Functional Functor, Composable Argument>
class Function : public Expression<Functor, Argument> {
public: 
    constexpr Function(Functor functor, typename Trait<Argument>::Reference argument)
    :   Expression<Functor, Argument>(functor, argument)
    ,   strides_(argument.shape())
    {}
    
    constexpr type dtype() const {
        return std::get<0>(this->operands).dtype();
    }

    constexpr Shape const& shape() const {
        return std::get<0>(this->operands).shape();
    }  
    
    constexpr Strides const& strides() const {
        return strides_;
    }
    
    auto offset() const {
        return 0;
    }
 
    Tensor forward(Context const& context) const {
        Tensor source = std::get<0>(this->operands).forward(context);
        Tensor target(dtype(), shape());
        this->operation(source, target);
        return target;
    } 

private:
    Strides strides_; 
};

} namespace tannic::function {

using tannic::expression::Function;

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
 * @brief Functor inverse square root (1/√x)
 * Applies element-wise inverse square root to tensor elements
 */
struct Rsqrt { 
    float epsilon = 0.0f;
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
template<Composable Operand>
constexpr auto log(Operand&& operand) {
    return Function<Log, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated exponential function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise e^(operand)
 * @see Exp
 */
template<Composable Operand>
constexpr auto exp(Operand&& operand) {
    return Function<Exp, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated square root expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise √operand
 * @note Returns NaN for negative inputs
 * @see Sqrt
 */
template<Composable Operand>
constexpr auto sqrt(Operand&& operand) {
    return Function<Sqrt, Operand>({}, std::forward<Operand>(operand));
}  

/**
 * @brief Creates a lazy-evaluated inverse square root expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise √operand
 * @note Returns NaN for negative inputs
 * @see Rsqrt
 */
template<Composable Operand>
constexpr auto rsqrt(Operand&& operand, float epsilon = 0.0f) {
    return Function<Rsqrt, Operand>({epsilon}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated absolute value expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise |operand|
 * @see Abs
 */
template<Composable Operand>
constexpr auto abs(Operand&& operand) {
    return Function<Abs, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated sine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise sin(operand)
 * @see Sin
 */
template<Composable Operand>
constexpr auto sin(Operand&& operand) {
    return Function<Sin, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated cosine function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise cos(operand)
 * @see Cos
 */
template<Composable Operand>
constexpr auto cos(Operand&& operand) {
    return Function<Cos, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated tangent function expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression (in radians)
 * @return Functor expression representing element-wise tan(operand)
 * @note Returns NaN where cosine equals zero
 * @see Tan
 */
template<Composable Operand>
constexpr auto tan(Operand&& operand) {
    return Function<Tan, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic sine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise sinh(operand)
 * @see Sinh
 */
template<Composable Operand>
constexpr auto sinh(Operand&& operand) {
    return Function<Sinh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic cosine expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise cosh(operand)
 * @see Cosh
 */
template<Composable Operand>
constexpr auto cosh(Operand&& operand) {
    return Function<Cosh, Operand>({}, std::forward<Operand>(operand));
}

/**
 * @brief Creates a lazy-evaluated hyperbolic tangent expression
 * @tparam Operand Type satisfying Expression concept
 * @param operand Input tensor expression
 * @return Functor expression representing element-wise tanh(operand) 
 * @see Tanh
 */
template<Composable Operand>
constexpr auto tanh(Operand&& operand) {
    return Function<Tanh, Operand>({}, std::forward<Operand>(operand));
}

} namespace tannic {
    
using function::log;
using function::exp;
using function::sqrt;
using function::rsqrt;
using function::abs;
using function::sin;
using function::cos;
using function::tan;
using function::sinh;
using function::cosh;
using function::tanh;

} //namespace tannic

#endif // FUNCTIONS_HPP