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

#include "Concepts.hpp"
#include "Traits.hpp" 
#include "Shape.hpp"
#include "Strides.hpp"
#include "Tensor.hpp" 

namespace tannic {
namespace function {

template<class Functor, Expression Operand>
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

    auto offset() const {
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


template<Expression Operand>
constexpr auto log(Operand&& operand) {
    return Function<Log, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto exp(Operand&& operand) {
    return Function<Exp, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto sqrt(Operand&& operand) {
    return Function<Sqrt, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto abs(Operand&& operand) {
    return Function<Abs, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto sin(Operand&& operand) {
    return Function<Sin, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto cos(Operand&& operand) {
    return Function<Cos, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto tan(Operand&& operand) {
    return Function<Tan, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto sinh(Operand&& operand) {
    return Function<Sinh, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto cosh(Operand&& operand) {
    return Function<Cosh, Operand>({}, std::forward<Operand>(operand));
}

template<Expression Operand>
constexpr auto tanh(Operand&& operand) {
    return Function<Tanh, Operand>({}, std::forward<Operand>(operand));
}
} // namespace functions 

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