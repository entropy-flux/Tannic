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

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <type_traits>
#include <initializer_list>
#include <variant>
#include <optional>
#include <iostream>
#include <vector>
#include <list>

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp"
#include "Storage.hpp" 
 
class Tensor {
public:
    using index_type = int;
    using size_type = Shape::size_type;
    using rank_type = Shape::rank_type;
    using difference_type = std::ptrdiff_t;

    type dtype() const { return dtype_; }
    Shape const& shape() const { return shape_; }
    Strides const& strides() const { return strides_; }
    Storage const & storage() const { return storage_; }
    auto size() const { return shape_.size(); }      
    auto size(Shape::index_type index) const { return shape_[index]; } 
    auto stride(Strides::index_type index) const { return strides_[index]; }
    auto rank() const { return shape_.rank(); } 
    std::byte* address() { return static_cast<std::byte*>(storage_.address()) + offset_; } 
    std::byte const* address() const { return static_cast<std::byte const*>(storage_.address()) + offset_;   }  
    Tensor() = default;
 
    template<class Size>
    Tensor(type dtype, std::initializer_list<Size> sizes, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(sizes.begin(), sizes.end()) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}
     
    Tensor(type dtype, Shape const& shape, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}

    Tensor(type dtype, Shape&& shape, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(std::move(shape)) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
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
 
    const Tensor& forward() const {
        return *this;
    }


    constexpr index_type normalize(index_type index) const {  
        auto size = shape_.front();
        if (index < 0) index += size;
        assert(index >= 0 && index < size && "Index out of bound");
        return index;
    }

    Tensor operator[](index_type index) const { 
        assert(rank() >= 1 && "Cannot slice scalar tensor");
        difference_type offset = offset_ + normalize(index) * strides_.front() * dsizeof(dtype_);
        Strides strides(strides_.begin() + 1, strides_.end());
        Shape shape(shape_.begin() + 1, shape_.end());
        return Tensor(storage_, std::move(shape), std::move(strides), dtype_, offset);
    }

    Tensor transpose(Shape::index_type first, Shape::index_type second) const {
        return Tensor(storage_, std::move(shape_.transpose(first, second)), std::move(strides_.transpose(first, second)), dtype_, offset_);
    }

    Tensor squeeze() const {
        std::vector<Shape::size_type> shape;
        std::vector<Strides::step_type> strides;
        for (rank_type dimension = 0; dimension < shape_.rank(); ++dimension) {
            if (shape_[dimension] != 1) {
                shape.push_back(shape_[dimension]);
                strides.push_back(strides_[dimension]);
            }
        }

        if (shape.empty()) {
            shape.push_back(1);
            strides.push_back(1);
        } 
         
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }
  
    Tensor unsqueeze(Shape::index_type index) const {
        std::vector<Shape::size_type> shape(shape_.begin(), shape_.end());
        std::vector<Strides::step_type> strides(strides_.begin(), strides_.end()); 
        { auto iterator = shape.begin(); std::advance(iterator, shape_.normalize(index, 1)); shape.insert(iterator, 1); }
        { auto iterator = strides.begin(); std::advance(iterator, strides_.normalize(index, 1)); strides.insert(iterator, 1); }
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }


    template<typename... Indexes>
    Tensor unsqueeze(Indexes... indexes) const {
        Shape::index_type extra = sizeof...(indexes);
        std::vector<Shape::index_type> dimensions{static_cast<Shape::index_type>(indexes)...};
        std::sort(dimensions.begin(), dimensions.end());

        std::vector<Shape::size_type> shape(shape_.begin(), shape_.end());
        std::vector<Strides::step_type> strides(strides_.begin(), strides_.end()); 
        for (Shape::index_type index : dimensions) {   
            { auto iterator = shape.begin(); std::advance(iterator, shape_.normalize(index, extra)); shape.insert(iterator, 1); }
            { auto iterator = strides.begin(); std::advance(iterator, strides_.normalize(index, extra)); strides.insert(iterator, 1); }
        }
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }   

    template<typename T>
    void operator=(T value) {
        assert(rank() == 0 && "Can't assign a scalar to a tensor with more than one element."); 
        assert(sizeof(value) == dsizeof(dtype_)); // TODO: add dcast to avoid errors with T != dtype.
        storage_.copy(address(), &value, sizeof(value));
    }

    template<typename T> 
    bool operator==(T value) const {
        assert(rank() == 0 && "Can't compare a scalar to an Array with more than one element.");
        assert(sizeof(value) == dsizeof(dtype_)); // TODO: add dcast to avoid errors with T != dtype.
        return storage_.compare(address(), &value, sizeof(value));
    }

    template<typename T>
    T item() const {
        return *reinterpret_cast<const T*>(address());
    }
 
    Tensor(const Storage& storage, Shape&& shape, Strides&& strides, type dtype, difference_type offset)
    :   storage_(storage)
    ,   shape_(std::move(shape))
    ,   strides_(std::move(strides))
    ,   dtype_(dtype)
    ,   offset_(offset)
    {}

    bool is_transposed() const {
        return strides_[-1] > strides_[-2] ? true : false;
    } 

    private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
    Storage storage_;
    difference_type offset_ = 0; 
}; 

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    if (tensor.rank() == 0) {  
        traits[tensor.dtype()].print(os, tensor.storage().address());
    } else {  
        os << "[";
        if (tensor.rank() == 1) {  
            for (auto index = 0; index < tensor.shape().front(); ++index) {
                if (index != 0) os << ", ";
                os << tensor[index];   
            }
        } else {  
            for (auto index = 0; index < tensor.shape().front(); ++index) {
                if (index != 0) os << ",\n ";
                os << tensor[index];   
            }
        }
        os << "]";
    }
    return os;
}

#endif // TENSOR_HPP