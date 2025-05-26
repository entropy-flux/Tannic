#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include <string>
#include "Types.hpp"
#include "Shape.hpp"
#include "View.hpp"

class Parameter {
public:
    constexpr type dtype() const {
        return dtype_;
    }

    constexpr const Shape& shape() const {
        return shape_;
    }

    constexpr const char* name() const {
        return name_;
    }
 
    constexpr Parameter(const Shape& shape, type dtype, const char* name = "")
        : name_(name), dtype_(dtype), shape_(shape) {}

    const Parameter& forward() const {
        return *this;
    }

    View view() const {
        return View(shape_, dtype_);
    }

private:
    const char* name_;
    type dtype_;
    Shape shape_;
};

#endif // PARAMETER_HPP
