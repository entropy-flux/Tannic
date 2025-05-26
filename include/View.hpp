#ifndef VIEW_HPP
#define VIEW_HPP
 
#include "Types.hpp"
#include "Shape.hpp"

class View {
public:
    type dtype() const {
        return dtype_;
    }

    const Shape& shape() const {
        return shape_;
    } 

    View(const Shape& shape, type dtype)
        : dtype_(dtype), shape_(shape) {}
  
    const View& forward() const {
        return *this;
    }

private:
    type dtype_;
    Shape shape_; 
}; 

#endif // VIEW_HPP