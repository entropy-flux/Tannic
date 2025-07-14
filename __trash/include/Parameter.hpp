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

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp"
#include "Storage.hpp"  
#include "Slices.hpp"

class Parameter {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t;  

    constexpr Parameter(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_)   
    {}

    constexpr Parameter(type dtype, Shape shape, Strides strides)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)   
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

    constexpr auto nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    constexpr auto rank() const { 
        return shape_.rank(); 
    }          

    template<class Index>
    constexpr auto operator[](Index index) const {  
        return expression::Slice<Parameter, Index>(*this, index);
    }
      
    constexpr auto operator[](Range index) const { 
        return expression::Slice<Parameter, Range>(*this, index);
    }

private:
    type dtype_;
    Shape shape_; 
    Strides strides_;  
};  

#endif