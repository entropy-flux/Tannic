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
#include <memory>

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp"
#include "Storage.hpp"  
#include "Slices.hpp"

class Tensor {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t;  

    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0)
    {}
 


    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor(const Expression& expression) {
        *this = expression.forward();  
    }

    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor& operator=(const Expression& expression) {  
        *this = expression.forward(); 
        return *this;
    }


public:
    void initialize(Allocator allocator = Host{}) {
        storage_ = std::make_shared<Storage>(shape_.size(), dsizeof(dtype_), allocator);
    }


public: 
    type dtype() const { 
        return dtype_; 
    }

    auto resource() const {
        return storage_-> resource();
    }

    Shape const& shape() const { 
        return shape_; 
    }

    Strides const& strides() const { 
        return strides_; 
    }

    std::ptrdiff_t offset() const {
        return offset_;
    }

    std::byte* address() {
        return static_cast<std::byte*>(storage_-> address()) + offset_;
    }

    std::byte const* address() const {
        return static_cast<std::byte const*>(storage_-> address()) + offset_;
    }

    auto nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    auto rank() const { 
        return shape_.rank(); 
    }          
 
    Tensor forward() const{
        return *this;
    }

    bool is_initialized() const {
        return !(storage_ == nullptr);
    }

public:  
    template<class Index>
    auto operator[](Index index) {   
        return expression::Slice<Tensor, Index>(*this, std::make_tuple(index));
    }

    auto operator[](Range index) { 
        return expression::Slice<Tensor, Range>(*this, std::make_tuple(index));
    }

    template<class ... Indexes>
    auto operator[](Indexes... indexes) {
        return expression::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    }

protected:
    template <class Expression, class... Indexes>
    friend class expression::Slice;

    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset, std::shared_ptr<Storage> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)  
    ,   storage_(std::move(storage)) 
    {}
 
    void assign(std::byte const* value, std::ptrdiff_t offset); 
    bool compare(std::byte const* value, std::ptrdiff_t offset) const;
     
private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
    std::shared_ptr<Storage> storage_;
    std::ptrdiff_t offset_;   
}; 

template <class Expression, class... Indexes>
Tensor expression::Slice<Expression, Indexes...>::forward() const {  
    Tensor source = expression.forward();
    return Tensor(dtype_, shape_, strides_, offset_, source.storage_);
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