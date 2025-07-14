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
#include <cassert>

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp" 
#include "Slices.hpp"
#include "Storage.hpp"
#include "Operations.hpp"

class Tensor {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t;  

    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0) { 
    }  

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
        storage_ = std::make_shared<Storage>(nbytes(), allocator);
    }


public: 
    type dtype() const { 
        return dtype_; 
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

    std::size_t nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    auto rank() const { 
        return shape_.rank(); 
    }          

    Tensor& forward() {
        return *this;
    } 

    Tensor const& forward() const{
        return *this;
    } 
  
    std::byte* buffer() const {
        return static_cast<std::byte*>(storage_->address()) + offset_;
    } 

public:
    bool is_initialized() const {
        return storage_ ? true : false;
    }

    auto resource() const {
        assert(storage_ && "Cannot get resource of an initializer tensor.");
        return storage_->resource();
    }

    Allocator const& allocator() const {
        assert(storage_ && "Cannot get resource of an initializer tensor.");
        return storage_->allocator();
    }
 

public:
    template<class Index>
    auto operator[](Index index) {   
        return view::Slice<Tensor, Index>(*this, std::make_tuple(index));
    }

    auto operator[](Range index) { 
        return view::Slice<Tensor, Range>(*this, std::make_tuple(index));
    }

    template<class ... Indexes>
    auto operator[](Indexes... indexes) {
        return view::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    }

protected:   
    template <class Expression, class... Indexes> 
    friend class view::Slice;
    
    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset, std::shared_ptr<Storage> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)   
    ,   storage_(std::move(storage))
    {}

    void assign(std::byte const*, std::ptrdiff_t); 
    bool compare(std::byte const*, std::ptrdiff_t) const;
  
     
private:
    type dtype_;
    Shape shape_; 
    Strides strides_; 
    std::ptrdiff_t offset_;   
    std::shared_ptr<Storage> storage_ = nullptr;
};  

template <class Expression, class... Indexes>
Tensor view::Slice<Expression, Indexes...>::forward() const {   
    Tensor source = expression.forward();
    return Tensor(dtype_, shape_, strides_, offset_, source.storage_);
}  

template<class Operation, class Operand>
Tensor operation::Unary<Operation, Operand>::forward() const { 
    Tensor result(dtype(), shape());  
    operation.forward(operand, result);
    return result;
}

template<class Operation, class Operand, class Cooperand>
Tensor operation::Binary<Operation, Operand, Cooperand>::forward() const {
    Tensor result(dtype(), shape());
    operation.forward(operand, cooperand, result);
    return result;
} 
  

#endif // TENSOR_HPP