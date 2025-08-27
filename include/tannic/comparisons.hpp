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

#ifndef COMPARISONS_HPP
#define COMPARISONS_HPP

#include "concepts.hpp"
#include "shape.hpp"
#include "tensor.hpp"

namespace tannic::expression {

template<class Criteria, Expression First, Expression Second>
class Comparison {
public:  
    Criteria criteria;
    typename Trait<Expression>::Reference first;
    typename Trait<Expression>::Reference second;

    constexpr Comparison(Criteria criteria, typename Trait<Expression>::Reference first, typename Trait<Expression>::Reference second) 
    :   criteria(criteria)
    ,   first(first)
    ,   second(second)
    ,   shape_(first.shape())
    ,   strides_(shape_)
    {
        if(first.shape() != second.shape()) 
            throw Exception("Cannot compare tensors of different shape"); 
    }

    constexpr type dtype() const {
        return boolean;
    }

    constexpr Shape const shape() const {
        return shape_;
    }

    constexpr Strides strides() const {
        return strides_;
    }

    constexpr std::ptrdiff_t offset() const {
        return 0;
    }
    
    Tensor forward() const {
        Tensor result(boolean, shape_, strides_, 0);
        criteria.forward(first, second, result);
        return result;
    }


private:
    Shape shape_;
    Strides strides_;

};

struct EQ {
    void forward(Tensor const&, Tensor const&, Tensor&);
};

struct NE { 
    void forward(Tensor const&, Tensor const&, Tensor&);
};

struct GT {
    void forward(Tensor const&, Tensor const&, Tensor&); 
};

struct GE {
    void forward(Tensor const&, Tensor const&, Tensor&); 
};

struct LT {
    void forward(Tensor const&, Tensor const&, Tensor&); 
};

struct LE { 
    void forward(Tensor const&, Tensor const&, Tensor&);
};

} namespace tannic {


}


#endif // COMPARISONS_HPP