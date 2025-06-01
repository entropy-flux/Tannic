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
