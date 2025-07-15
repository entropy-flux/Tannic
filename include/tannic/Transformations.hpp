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

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include <tuple>
#include <array>
#include <vector>

#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp" 
#include "Tensor.hpp"
#include "Concepts.hpp"

namespace tannic {

class Tensor;

namespace expression {    
  
template<class Operation, Operable ... Operands>
class Transformation {
public:
    Operation operation;
    std::tuple<typename Trait<Operands>::Reference...> operands;

    constexpr Transformation(Operation operation, typename Trait<Operands>::Reference... operands)
    :   operation(operation)
    ,   operands(operands...) 
    ,   dtype_(operation.promote(operands.dtype()...))
    ,   shape_(operation.broadcast(operands.shape()...)) 
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

    constexpr std::ptrdiff_t offset() const {
        return 0;
    }

    auto forward() const -> decltype(auto) {
        return std::apply([&](const auto&... arguments) {
            return operation.forward(dtype_, shape_, arguments.forward()...);
        }, operands);
    }

private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
}; 

static constexpr auto index(type inner, type outer) {
    return static_cast<int>(inner) + static_cast<int>(outer) * static_cast<int>(TYPES);
} 

struct Composition { 
    void forward(Tensor const&, Tensor const&, Tensor&) const;

 
    static constexpr auto promotions = []() {
        std::array<type, index(TYPES, TYPES)> table{};  
        table[index(int8, int8)]   = int32;
        table[index(int8, int16)]  = int32;
        table[index(int8, int32)]  = int32;
        table[index(int8, int64)]  = int64;  

        table[index(int16, int8)]    = int32;
        table[index(int16, int16)]   = int32;
        table[index(int16, int32)]   = int32;
        table[index(int16, int64)]   = int64; 
        
        table[index(int32, int8)]    = int32;
        table[index(int32, int16)]   = int32;
        table[index(int32, int32)]   = int64;
        table[index(int32, int64)]   = int64; 
        
        table[index(int64, int8)]    = int64;
        table[index(int64, int16)]   = int64;
        table[index(int64, int32)]   = int64;
        table[index(int64, int64)]   = int64; 
        
        table[index(int32, float32)] = float32;
        table[index(float32, int32)] = float32;
        table[index(int32, float64)] = float64;
        table[index(float64, int32)] = float64; 
        
        table[index(float32, float32)] = float32;
        table[index(float32, float64)] = float64;
        table[index(float64, float32)] = float64;
        table[index(float64, float64)] = float64; 
        return table;
    }();  

    Tensor forward(type dtype, Shape const& shape, Tensor const& outer, Tensor const& inner) const {
        Tensor result(dtype, shape);
        forward(outer, inner, result);
        return result;
    }

    static constexpr type promote(type inner, type outer) {
        return promotions[index(inner, outer)];
    }

    static constexpr Shape broadcast(Shape const& first, Shape const& second) {
        auto first_rank = first.rank();
        auto second_rank = second.rank();
         
        if (first_rank == 1 && second_rank == 1) {
            assert(first[0] == second[0] && "Vector dimensions must match for dot product");
            return Shape{};  // scalar result
        }
         
        if (first_rank == 1 && second_rank == 2) {
            assert(first[0] == second[1] && "Matrix inner dimensions do not match");
            return Shape{second[0]};
        }
         
        if (first_rank == 2 && second_rank == 1) {
            assert(first[1] == second[0] && "Matrix inner dimensions do not match");
            return Shape{first[0]};
        } 
         
        assert(first_rank >= 2 && second_rank >= 2 && "Inputs must have rank >= 2");
        
        Shape first_batches(first.begin(), first.end() - 2);
        Shape second_batches(second.begin(), second.end() - 2);
        Shape batches = operation::broadcast(first_batches, second_batches);
        
        auto K1 = *(first.end() - 1);
        auto K2 = *(second.end() - 2);   
        assert(K1 == K2 && "Inner dimensions must match for matmul");
        
        auto M = *(first.end() - 2);
        auto N = *(second.end() - 1);
        
        std::vector<Shape::size_type> result(batches.begin(), batches.end());
        result.push_back(M);
        result.push_back(N);
        return Shape(result);
    }
};


template<Operable Outer, Operable Inner>
constexpr auto composition(Outer&& outer, Inner&& inner) {
    return Transformation<Composition, std::decay_t<Outer>, std::decay_t<Inner>>{
        {}, std::forward<Outer>(outer), std::forward<Inner>(inner)
    };
}

} // namespace expression

template<Operable Multiplicand, Operable Multiplier>
constexpr auto matmul(Multiplicand&& multiplicand, Multiplier&& multiplier) {
    return expression::composition(
        std::forward<Multiplicand>(multiplicand),
        std::forward<Multiplier>(multiplier)
    );
}

} // namespace tannic

#endif // TRANSFORMATIONS_HPP