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

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>  

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp"
#include "Storage.hpp"  
#include "Slices.hpp"

class Tensor {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t; 
    using difference_type = std::ptrdiff_t;  

    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0)
    ,   storage_{} 
    ,   is_contiguous_(true)
    {}
 
    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor(const Expression& expression) 
    :   dtype_(expression.dtype())
    ,   shape_(expression.shape())
    ,   strides_(expression.strides())
    ,   offset_(expression.offset())   
    {}

    std::byte* data() { 
        return static_cast<std::byte*>(storage_.data()) + offset_; 
    } 

    std::byte const* data() const { 
        return static_cast<std::byte const*>(storage_.data()) + offset_;   
    }  
 
    type dtype() const { 
        return dtype_; 
    }

    Shape const& shape() const { 
        return shape_; 
    }

    Strides const& strides() const { 
        return strides_; 
    }

    difference_type offset() const {
        return offset_;
    }

    auto nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    auto rank() const { 
        return shape_.rank(); 
    }          

    template<class Index>
    auto operator[](Index index) const {  
        return expression::Slice<Tensor, Index>(*this, index);
    }

    auto operator[](Range index) const { 
        return expression::Slice<Tensor, Range>(*this, index);
    }

    Tensor& forward() {
        return *this;
    }

    Tensor const& forward() const {
        return *this;
    }

protected:
    template <class Expression, class... Indexes>
    friend class expression::Slice;

    Tensor(type dtype, Shape shape, Strides strides, difference_type offset, bool is_contiguous)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)  
    ,   storage_{} 
    ,   is_contiguous_(is_contiguous)
    {}
     
private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
    Storage storage_;
    difference_type offset_;  
    bool is_contiguous_; 
}; 

template <class Expression, class... Indexes>
Tensor expression::Slice<Expression, Indexes...>::forward() const { 
    return Tensor(dtype_, shape_, strides_, offset_, false);
} 

template <class Expression, class... Indexes>
template <typename T>
void expression::Slice<Expression, Indexes...>::operator=(T value) {

}

template <class Expression, class... Indexes>
template <typename T>
bool expression::Slice<Expression, Indexes...>::operator==(T value) {
    
}

#endif // TENSOR_HPP