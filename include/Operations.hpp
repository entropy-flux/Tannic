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
#include "Types.hpp"
#include "Shape.hpp"
#include "Expressions.hpp"

class Tensor;
  
enum class Arity : uint8_t { Unary, Binary };
template<class Operator, Arity arity>  struct Operation;

template<class Operator>
struct Operation<Operator, Arity::Unary> {
    template<class Type>
    static constexpr type promote(Type dtype) { return dtype; }

    template<class Shape>
    static constexpr Shape const& broadcast(Shape const& shape) { return shape; }

    template<class Tensor>
    Tensor forward(const Tensor& operand) const{
        Tensor result(operand.dtype(), operand.shape());
        static_cast<const Operator*>(this)->forward(operand, result); 
        return result;
    };
};
 
template<class Operator>
struct Operation<Operator, Arity::Binary> { 

    template<class Type>
    static constexpr Type promote(Type first, Type second) { 
        return static_cast<uint8_t>(first) > static_cast<uint8_t>(second) ? first : second; 
    }

    template<class Shape>
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
    
    template<class Tensor>
    Tensor forward(Tensor const& operand, Tensor const& cooperand) const {  
        Tensor result(promote(operand.dtype(), cooperand.dtype()), broadcast(operand.shape(), cooperand.shape()));
        static_cast<const Operator*>(this)->forward(operand, cooperand, result); 
        return result;
    }
};

namespace symbol {
 
struct Negation : public Operation<Negation, Arity::Unary> { 
    using Operation<Negation, Arity::Unary>::forward; 
    void forward(Tensor const&, Tensor&) const; 
};

struct Addition : Operation<Addition, Arity::Binary> { 
    using Operation<Addition, Arity::Binary>::forward; 
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

struct Subtraction : Operation<Subtraction, Arity::Binary>  { 
    using Operation<Subtraction, Arity::Binary>::forward;
    void forward(Tensor const&, Tensor const&,  Tensor&) const; 
};

struct Multiplication : Operation<Multiplication, Arity::Binary>  { 
    using Operation<Multiplication, Arity::Binary>::forward;
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

} // symbol
  
template<class Operand>
constexpr auto operator-(Operand const & operand) {
    return Unary<symbol::Negation, Operand>{{}, operand};
}

template<class Augend, class Addend>
constexpr auto operator+(Augend const& augend, Addend const& addend) {
    return Binary<symbol::Addition, Augend, Addend>{{}, augend, addend};
} 

template<class Subtrahend , class Minuend>
constexpr auto operator-(Subtrahend const& subtrahend, Minuend const& minuend) {
    return Binary<symbol::Subtraction, Subtrahend, Minuend>{{}, subtrahend, minuend};
} 

template<class Multiplicand, class Multiplier>
constexpr auto operator*(Multiplicand const& multiplicand, Multiplier const& multiplier) {
    return Binary<symbol::Multiplication, Multiplicand, Multiplier>{{}, multiplicand, multiplier};
}  

#endif // OPERATIONS_HPP