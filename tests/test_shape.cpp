#include <gtest/gtest.h>
#include "Shape.hpp" 

TEST(Test, Constructor) {     
    constexpr Shape const_shape(1,2,3);
    Shape iterable_shape(const_shape);
    ASSERT_EQ(iterable_shape[0], 1);
    ASSERT_EQ(iterable_shape[1], 2);
    ASSERT_EQ(iterable_shape[2], 3); 
    ASSERT_EQ(iterable_shape, const_shape); 
    constexpr Shape shape1(1,2,3,4,5);;
    constexpr Shape shape2(1,2,3,4,5);;
    static_assert(shape1.rank() == 5);
    static_assert(shape2[0] == 1);
    static_assert(shape1 == shape2); 
}