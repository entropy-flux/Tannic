#include <gtest/gtest.h>
#include "Parameter.hpp"
#include "Expressions.hpp"
#include "Shape.hpp"
#include "Operations.hpp"
#include "Tensor.hpp"

struct Dummy : public Operation {
    void perform(const View& operand, Tensor& result) const {}
};

TEST(Test, Constexpr) {
    constexpr Parameter tensor(Shape(1,2,3,4), float32); 
    Tensor result = expression::Unary<Dummy, Parameter>({}, tensor);
    ASSERT_EQ(result.dtype(), float32);
    ASSERT_EQ(result.shape(), Shape(1,2,3,4)); 
}

TEST(Test, ConstexprBroadcast) {
    constexpr Parameter x(Shape(1,2,3,1), float32);
    constexpr Parameter y(Shape(1,2,1,4), float32);
    constexpr Parameter z(Shape(4,2,3,4), float32);
    constexpr auto r = x * y + y * z;
    static_assert(r.shape() == Shape(4, 2, 3, 4));
}

constexpr auto somefunc(Parameter const& x, Parameter const& y, Parameter const& z) { 
    return x * y + y * z;    
} 

TEST(Test, ConstexprFunctions) {
    constexpr Parameter x(Shape{3,1,3,4}, float32);
    constexpr Parameter y(Shape{1,1,3,1}, float32);
    constexpr Parameter z(Shape{1,1,1}, float32);
    constexpr auto r = somefunc(x, y, z);   
    static_assert(r.shape() == Shape(3, 1, 3, 4));
    Tensor rr = somefunc(x, y, z);   
    ASSERT_EQ(rr.shape(), Shape(3, 1, 3, 4));
}