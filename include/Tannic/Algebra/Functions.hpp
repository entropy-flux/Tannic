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
// This file is part of Tannic, a A C++ tensor library.

#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
 
#include "Expressions.hpp"

class Tensor;

namespace ta {

template<class Functor>
struct Function {
    static constexpr auto promote(auto dtype) noexcept { return dtype; } 
    static constexpr auto broadcast(const Shape& shape) noexcept { return shape; }

    template<class Tensor>
    Tensor forward(const Tensor& operand) const {
        Tensor result(operand.shape(), operand.dtype());
        static_cast<const Functor*>(this)->forward(operand, result);  
        return result;
    }
};
 
struct Log : public Function<Log> {  
    void forward(Tensor const&, Tensor&) const;
};

struct Exp : public Function<Exp> { 
    void forward(Tensor const&, Tensor&) const;
};

struct Sqrt : public Function<Sqrt> {
    void forward(Tensor const&, Tensor&) const;
};

struct Abs : public Function<Abs> {
    void forward(Tensor const&, Tensor&) const;
};

struct Sin : public Function<Sin> {
    void forward(Tensor const&, Tensor&) const;
};

struct Sinh : public Function<Sinh> {
    void forward(Tensor const&, Tensor&) const;
};

struct Cos : public Function<Cos> {
    void forward(Tensor const&, Tensor&) const;
};

struct Cosh : public Function<Cosh> {
    void forward(Tensor const&, Tensor&) const;
};

struct Tan : public Function<Tan> {
    void forward(Tensor const&, Tensor&) const;
};

struct Tanh : public Function<Tanh> {
    void forward(Tensor const&, Tensor&) const;
}; 
 
} //ta

template<class Operand>
constexpr auto log(Operand const& operand) {
    return Unary<ta::Function<ta::Log>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto exp(Operand const& operand) {
    return Unary<ta::Function<ta::Exp>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sqrt(Operand const& operand) {
    return Unary<ta::Function<ta::Sqrt>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto abs(Operand const& operand) {
    return Unary<ta::Function<ta::Abs>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sin(Operand const& operand) {
    return Unary<ta::Function<ta::Sin>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sinh(Operand const& operand) {
    return Unary<ta::Function<ta::Sinh>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto cos(Operand const& operand) {
    return Unary<ta::Function<ta::Cos>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto cosh(Operand const& operand) {
    return Unary<ta::Function<ta::Cosh>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto tan(Operand const& operand) {
    return Unary<ta::Function<ta::Tan>, Operand>{{}, operand};
}

template<class Operand>
constexpr auto tanh(Operand const& operand) {
    return Unary<ta::Function<ta::Tanh>, Operand>{{}, operand};
}
  
#endif // FUNCTIONS_HPP