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

#ifndef LIMITS_HPP
#define LIMITS_HPP

#include "types.hpp"
#include "shape.hpp"
#include "strides.hpp"

namespace tannic {

class Tensor;

} namespace tannic::expression {

class Zero {
public:
    constexpr Zero(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape)
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
  
    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};

class One {
public:
    constexpr One(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape)
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
    
    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};

enum class Sign {
    Positive,
    Negative
};

template<Sign Sign>
class Infinity {
public:
    constexpr Infinity(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape)
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
    
    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};

} namespace tannic {

constexpr auto zeros(type dtype, Shape const& shape) {
    return expression::Zero(dtype, shape);
}

constexpr auto ones(type dtype, Shape const& shape) {
    return expression::One(dtype, shape);
}

constexpr auto infinity(type dtype, Shape const& shape) {
    return expression::Infinity<expression::Sign::Positive>(dtype, shape);
}

}

#endif // LIMITS_HPP