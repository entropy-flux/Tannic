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

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include <string>
#include "Types.hpp"
#include "Shape.hpp"
#include "Tensor.hpp"

class Parameter {
public:
    constexpr type dtype() const { return dtype_; }
    constexpr const Shape& shape() const { return shape_; }
    constexpr const char* name() const { return name_; }
 
    consteval Parameter(const Shape& shape, type dtype, const char* name = "")
    :   name_(name)
    ,   dtype_(dtype)
    ,   shape_(shape) {}

private:
    const char* name_;
    type dtype_;
    Shape shape_;
};

#endif // PARAMETER_HPP