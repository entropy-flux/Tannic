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

#include "Traits.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
#include "Tensor.hpp" 

namespace tannic {
namespace expression {

template<class Functor, Operable Operand>
class Function {
public:
    Functor functor;
    typename Trait<Operand>::Reference operand;

    constexpr Function(Functor functor, typename Trait<Operand>::Reference operand)
    :   functor(functor)
    ,   operand(operand)
    {}

    constexpr type dtype() const {
        return operand.dtype();
    }

    constexpr Shape const& shape() const {
        return operand.shape();
    }  

    constexpr Strides const& strides() const {
        return operand.strides();
    }

    constexpr auto offset() const {
        return operand.offset();
    }

    Tensor forward() const {
        Tensor source = operand.forward();
        Tensor target(dtype(), shape());
        functor(source, target);
        return target;
    } 
};
 
struct Log {
    void operator()(Tensor const&, Tensor&) const;
};

struct Exp {
    void operator()(Tensor const&, Tensor&) const;
};

struct Sqrt { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Abs { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Sin {
    void operator()(Tensor const&, Tensor&) const;
};

struct Cos { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Tan { 
    void operator()(Tensor const&, Tensor&) const;
};
 
struct Sinh { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Cosh { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Tanh { 
    void operator()(Tensor const&, Tensor&) const;
};
template<Operable Expression> 
constexpr auto log(Expression&& expression) {
    return Function<Log, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto exp(Expression&& expression) {
    return Function<Exp, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto sqrt(Expression&& expression) {
    return Function<Sqrt, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto abs(Expression&& expression) {
    return Function<Abs, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto sin(Expression&& expression) {
    return Function<Sin, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto cos(Expression&& expression) {
    return Function<Cos, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto tan(Expression&& expression) {
    return Function<Tan, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto sinh(Expression&& expression) {
    return Function<Sinh, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto cosh(Expression&& expression) {
    return Function<Cosh, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}

template<Operable Expression>
constexpr auto tanh(Expression&& expression) {
    return Function<Tanh, std::decay_t<Expression>>({}, std::forward<Expression>(expression));
}
} // namespace expression 

using expression::log;
using expression::exp;
using expression::sqrt;
using expression::abs;
using expression::sin;
using expression::cos;
using expression::tan;
using expression::sinh;
using expression::cosh;
using expression::tanh;

} //namespace tannic

#endif // FUNCTIONS_HPP