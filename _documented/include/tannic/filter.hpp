// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Tensor Library.
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


#ifndef FILTER_HPP
#define FILTER_HPP

#include "types.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "expressions.hpp"
#include "tensor.hpp"

namespace tannic::expression {

template<class Rule, Composable Operand>
class Filter : public Expression<Rule, Operand> {
public:
    constexpr Filter(Rule mask, Trait<Operand>::Reference operand) 
    :   Expression<Rule, Operand>(mask, operand)
    {}

    constexpr type dtype() const {
        return std::get<0>(this->operands).dtype();
    }

    constexpr Shape const& shape() const {
        return std::get<0>(this->operands).shape();
    }

    constexpr Strides const& strides() const {
        return std::get<0>(this->operands).strides();
    }

    constexpr std::ptrdiff_t offset() const {
        return 0;
    }

    Tensor forward(Context const& context) const {
        Tensor source = std::get<0>(this->operands).forward();
        Tensor target(source.dtype(), source.shape(), source.strides());
        this->operation.forward(source, target);
        return target;
    }
};

enum class Position { Upper, Lower };

struct Triangular { 
    int offset;
    Position position;
    void forward(Tensor const&, Tensor&) const;
};

} namespace tannic {
 
using expression::Position;

template<Composable Operand>
constexpr auto triangular(Operand&& operand, Position position = Position::Upper, int offset = 0) {
    return expression::Filter<expression::Triangular, Operand>{{offset, position}, operand};
}

}


#endif // FILTER_HPP