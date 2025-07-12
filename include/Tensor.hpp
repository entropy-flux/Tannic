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
 
public:
    void initialize(Allocator allocator = Host{}) const {
        storage_ = Storage(shape_.size(), dsizeof(dtype_), allocator);
    }


public: 
    type dtype() const { 
        return dtype_; 
    }

    auto resource() const {
        return storage_.resource();
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

    std::byte* address() {
        return static_cast<std::byte*>(storage_.address()) + offset_;
    }

    std::byte const* address() const {
        return static_cast<std::byte*>(storage_.address()) + offset_;
    }

    auto nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    auto rank() const { 
        return shape_.rank(); 
    }          
 
    Tensor& forward() {
        return *this;
    }

    Tensor const& forward() const {
        return *this;
    }

    bool is_initialized() const {
        return !(storage_.address() == nullptr);
    }

public:
    template<class Index>
    auto operator[](Index index) const {   
        return expression::Slice<Tensor, Index>(*this, index);
    }

    auto operator[](Range index) const { 
        return expression::Slice<Tensor, Range>(*this, index);
    }

    template<class ... Indexes>
    auto operator[](Indexes... indexes) const {
        return expression::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
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
 
    void assign(std::byte const* value, difference_type offset); 
    bool compare(std::byte const* value, difference_type offset) const;
     
private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
    mutable Storage storage_;
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
    auto bytes = [](auto const& reference) -> std::byte const* {
        return reinterpret_cast<std::byte const*>(&reference);
    };

    switch (dtype_) {
        case int8: {
            int8_t casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        }
        case int16: {
            int16_t casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        }
        case int32: {
            int32_t casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        }
        case int64: {
            int64_t casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        }
        case float32: {
            float casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        }
        case float64: {
            double casted = value;
            expression.assign(bytes(casted), offset_);
            break;
        } 
        default:
            break;
    }
}

template <class Expression, class... Indexes>
template <typename T>
bool expression::Slice<Expression, Indexes...>::operator==(T value) const {  
    assert(rank() == 0 && "Cannot compare a scalar with a non scalar tensor");
    
    auto bytes = [](auto const& reference) -> std::byte const* {
        return reinterpret_cast<std::byte const*>(&reference);
    };

    switch (dtype_) {
        case int8: {
            int8_t casted = value;
            return expression.compare(bytes(casted), offset_); 
        }
        case int16: {
            int16_t casted = value;
            return expression.compare(bytes(casted), offset_); 
        }
        case int32: {
            int32_t casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case int64: {
            int64_t casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case float32: {
            float casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case float64: {
            double casted = value;
            return expression.compare(bytes(casted), offset_);
        } 
        default:
            return false;
        }
}
#endif // TENSOR_HPP