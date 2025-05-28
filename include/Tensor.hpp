#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <type_traits>
#include <initializer_list>
#include <variant>
#include <optional>
#include <iostream>
#include <vector>

#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp"
#include "Storage.hpp" 

class Tensor {
public:
    using size_type = Shape::size_type;
    using rank_type = Shape::rank_type;
    using difference_type = std::ptrdiff_t;

    type dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    const Strides& strides() const {return strides_; }
    rank_type rank() const { return shape_.rank(); }
    void* address() { return static_cast<std::byte*>(storage_.address()) + offset_; } 
    const void* address() const { return static_cast<const std::byte*>(storage_.address()) + offset_;   }

    Tensor(std::initializer_list<Shape::size_type> sizes, type dtype, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(sizes) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}

    Tensor(Shape&& shape, type dtype, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(shape) 
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

    Tensor operator[](int index) const {
        size_type size = shape_.front();
        if(index < 0) 
            index = size + index;
            
        assert(index < size && "Index out of range."); 
        difference_type offset = offset_ + index * strides_.front() * dsizeof(dtype_);
        Strides strides(strides_.begin() + 1, strides_.end());
        Shape shape(shape_.begin() + 1, shape_.end());
        return Tensor(storage_, std::move(shape), std::move(strides), dtype_, offset);
    }


    template<typename T>
    void operator=(T value) {
        assert(rank() == 0 && "Can't assign a scalar to an Array with more than one element.");
        traits[dtype_].assign(address(), value);
    }

    template<typename T> 
    bool operator==(T value) const {
        assert(rank() == 0 && "Can't compare a scalar to an Array with more than one element.");
        return traits[dtype_].compare(address(), value); 
    }

    template<typename T>
    T item() const {
        return std::any_cast<T>(traits[dtype_].retrieve(address()));
    }

    protected:
    Tensor(const Storage& storage, Shape&& shape, Strides&& strides, type dtype, difference_type offset)
    :   storage_(storage)
    ,   shape_(std::move(shape))
    ,   strides_(std::move(strides))
    ,   dtype_(dtype)
    ,   offset_(offset)
    {}

    private:
    type dtype_;
    Shape shape_; 
    Strides strides_;
    Storage storage_;
    difference_type offset_ = 0;
}; 

#endif // TENSOR_HPP