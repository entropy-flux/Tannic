// Copyright 2025 Eric Cardozo
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
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>  
#include <memory>
#include <cassert> 
#include <utility> 
 
#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp" 
#include "Storage.hpp" 
#include "Slices.hpp" 
#include "Views.hpp"
#include "Operations.hpp" 

namespace tannic { 

class Tensor {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t;  

    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0) {}  

    Tensor(type dtype, Shape shape, Strides strides)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides) 
    ,   offset_(0) {}  

    template <Expression Expression>
    Tensor(const Expression& expression) {
        *this = expression.forward(); 
    } 

    template <Expression Expression>
    Tensor& operator=(const Expression& expression) {
        *this = expression.forward(); 
        return *this;
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
        assert(is_initialized() && "Cannot perform computations with uninitialized tensors.");
        return *this;
    } 

    Tensor const& forward() const {
        assert(is_initialized() && "Cannot perform computations with uninitialized tensors."); 
        return *this;
    }
 
public: 
    void initialize(Allocator allocator = Host{}) {
        storage_ = std::make_shared<Storage>(nbytes(), allocator);
    }  

    std::byte* buffer() const {
        return static_cast<std::byte*>(storage_->address()) + offset_;
    } 
    
    bool is_initialized() const {
        return storage_ ? true : false;
    }
  
    auto environment() const {
        assert(storage_ && "Cannot get resource of an initializer tensor.");
        return storage_->environment();
    }

    Allocator const& allocator() const {
        assert(storage_ && "Cannot get resource of an initializer tensor.");
        return storage_->allocator();
    }  
   
public:
    template<Integral Index>
    auto operator[](Index index) const {   
        return expression::Slice<Tensor, Index>(*this, std::make_tuple(index));
    }

    auto operator[](indexing::Range range) const { 
        return expression::Slice<Tensor, indexing::Range>(*this, std::make_tuple(range));
    }

    template<class ... Indexes>
    auto operator[](Indexes... indexes) const {
        return expression::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    }
 
    auto transpose(int first = -1, int second = -2) const {
        return expression::Transpose<Tensor>(*this, std::make_pair<int, int>(std::move(first), std::move(second)));
    }

protected:   
    template <Expression Source, class... Indexes> 
    friend class expression::Slice;

    template <Expression Source> 
    friend class expression::Transpose;

    template <Expression Source>
    friend class expression::Reshape;
    
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

std::ostream& operator<<(std::ostream& ostream, Tensor tensor);  
 
template<Expression Source> 
inline std::ostream& operator<<(std::ostream& ostream, Source source) {
    Tensor tensor = source.forward();  
    ostream << tensor;
    return ostream;
} 

template<class Operation, Expression Operand>
Tensor operation::Unary<Operation, Operand>::forward() const { 
    Tensor result(dtype(), shape(), strides());  
    operation.forward(operand, result);
    return result;
}

template<class Operation, Expression Operand, Expression Cooperand>
Tensor operation::Binary<Operation, Operand, Cooperand>::forward() const {
    Tensor result(dtype(), shape(), strides());
    operation.forward(operand, cooperand, result);
    return result;
}  

template<Expression Source>
Tensor expression::Reshape<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.storage_);
}
 
template<Expression Source>
Tensor expression::Transpose<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.storage_);
}   

template<Expression Source, class... Indexes>
Tensor expression::Slice<Source, Indexes...>::forward() const {   
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source_.storage_);
}       

} // namespace tannic

#endif // TENSOR_HPP