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

static constexpr inline type promote(type first, type second) {
    return static_cast<int>(first) > static_cast<int>(second) ?  first : second;
}

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

template<class Operation, Expression Operand>
class Unary { 
public:
    Operation operation;
    Trait<Operand>::Reference operand;

    constexpr Unary(Operation operation, Trait<Operand>::Reference operand)
    :   operation(operation)
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

    std::ptrdiff_t offset() const {
        return 0;
    }

    Tensor forward() const;
};

template<class Operation, Expression Operand, Expression Cooperand>
class Binary {
public:
    Operation operation;
    Trait<Operand>::Reference operand;
    Trait<Cooperand>::Reference cooperand;

    constexpr Binary(Operation operation, Trait<Operand>::Reference operand, Trait<Cooperand>::Reference cooperand)
    :   operation(operation)
    ,   operand(operand)
    ,   cooperand(cooperand)
    ,   dtype_(promote(operand.dtype(), cooperand.dtype()))
    ,   shape_(broadcast(operand.shape(), cooperand.shape()))
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

    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
}; 

struct Negation {
    void forward(Tensor const&, Tensor&) const;  
};

struct Addition {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

struct Multiplication {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

struct Subtraction {
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};


template<Expression Operand>
constexpr auto operator-(Operand&& operand) {
    return Unary<Negation, Operand>{{}, std::forward<Operand>(operand)};
}
 
template<Expression Augend, Expression Addend>
constexpr auto operator+(Augend&& augend, Addend&& addend) {
    return Binary<Addition, Augend, Addend>{{}, std::forward<Augend>(augend), std::forward<Addend>(addend)};
}
 
template<Expression Subtrahend, Expression Minuend>
constexpr auto operator-(Subtrahend&& subtrahend, Minuend&& minuend) {
    return Binary<Subtraction, Subtrahend, Minuend>{{}, std::forward<Subtrahend>(subtrahend), std::forward<Minuend>(minuend)};
}
 
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