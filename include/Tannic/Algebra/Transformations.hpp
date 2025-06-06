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
// This file is part of Tannic, a machine learning tensor library for C++.

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "Algebra/Operations.hpp" 
 
class Tensor;

namespace ta {

struct Linear { 

    static constexpr Shape broadcast(Shape const& first, Shape const& second) {
        auto first_rank = first.rank();
        auto second_rank = second.rank();
 
        if (second_rank == 1) {
            throw std::invalid_argument("Vector not supported as transform matrix.");
        }
  
        if (first_rank == 1 && second_rank == 2) {
            if (first[0] != second[1])
                throw std::invalid_argument("Matrix inner dimensions do not match.");
            return Shape{second[0]};
        }
 
        Shape first_batches(first.begin(), first.end() - 2);
        Shape second_batches(second.begin(), second.end() - 2);
        Shape batches = Operation<Linear, Arity::Binary>::broadcast(first_batches, second_batches);
 
        auto K1 = *(first.end() - 1);    
        auto K2 = *(second.end() - 1);     
        if (K1 != K2)
            throw std::invalid_argument("Inner dimensions must match for linear.");

        auto M = *(first.end() - 2);
        auto N = *(second.end() - 2);  

        std::vector<Shape::size_type> result(batches.begin(), batches.end());
        result.push_back(M);
        result.push_back(N);
        return Shape(result);
    } 

    
    template<class Tensor, class Transform>
    Tensor forward(Tensor const& operand, Transform const& transform) const {   
        Tensor result(broadcast(operand.shape(), transform.shape()), promote(operand.dtype(), transform.dtype()));
        forward(operand, transform.transpose(-1, -2), result);       
        return result;
    }

    void forward(Tensor const&, Tensor const&, Tensor&) const; 
    
};
 
} // ta

template<class Argument, class Transform>
constexpr auto linear(Argument const& argument, Transform const& transform) {
    return Binary<ta::Linear, Argument, Transform>{{}, argument, transform};
}
 
#endif // TRANSFORMATIONS_HPP