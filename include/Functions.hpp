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

namespace expression {

template<class Functor, class Operand>
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

template<class Expression> 
constexpr auto log(Expression const& expression) {
    return Function<Log, Expression>({}, expression);
}

template<class Expression>
constexpr auto exp(Expression const& expression) {
    return Function<Exp, Expression>({}, expression);
}

template<class Expression>
constexpr auto sqrt(Expression const& expression) {
    return Function<Sqrt, Expression>({}, expression);
}

template<class Expression>
constexpr auto abs(Expression const& expression) {
    return Function<Abs, Expression>({}, expression);
}

template<class Expression>
constexpr auto sin(Expression const& expression) {
    return Function<Sin, Expression>({}, expression);
}

template<class Expression>
constexpr auto cos(Expression const& expression) {
    return Function<Cos, Expression>({}, expression);
}

template<class Expression>
constexpr auto tan(Expression const& expression) {
    return Function<Tan, Expression>({}, expression);
}

template<class Expression>
constexpr auto sinh(Expression const& expression) {
    return Function<Sinh, Expression>({}, expression);
}

template<class Expression>
constexpr auto cosh(Expression const& expression) {
    return Function<Cosh, Expression>({}, expression);
}

template<class Expression>
constexpr auto tanh(Expression const& expression) {
    return Function<Tanh, Expression>({}, expression);
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

#endif // FUNCTIONS_HPP