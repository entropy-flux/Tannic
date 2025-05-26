#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <type_traits>
#include <initializer_list>

#include "Types.hpp"
#include "Shape.hpp"
#include "View.hpp" 


class Tensor {
public:
    type dtype() const {
        return dtype_;
    }

    const Shape& shape() const {
        return shape_;
    }

    Tensor(std::initializer_list<Shape::size_type> sizes, type dtype)
        : dtype_(dtype), shape_(sizes) {}

    Tensor(const Shape& shape, type dtype)
        : dtype_(dtype), shape_(shape) {}

    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor(const Expression& expression) { 
        *this = expression.forward();
    }

    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor& operator=(const Expression& expression) {
        return expression.forward();
    }
 
    const Tensor& forward() const {
        return *this;
    }

    View view() const {
        return View(shape_, dtype_);
    }

private:
    type dtype_;
    Shape shape_; 
};

#endif // TENSOR_HPP