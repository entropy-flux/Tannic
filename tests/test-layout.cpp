#include <gtest/gtest.h>
#include "Shape.hpp" 
#include "Strides.hpp" 

using namespace tannic;

TEST(Test, Constructor) {     
    constexpr Shape const_shape(1,2,3);
    static_assert(const_shape.size() == 6);
    static_assert(const_shape.rank() == 3);
    Shape iterable_shape(const_shape);
    ASSERT_EQ(iterable_shape[0], 1);
    ASSERT_EQ(iterable_shape[1], 2);
    ASSERT_EQ(iterable_shape[2], 3); 
    ASSERT_EQ(iterable_shape, const_shape); 
    constexpr Shape shape1(1,2,3,4,5);;
    static_assert(shape1.size() == 1*2*3*4*5);
    constexpr Shape shape2(1,2,3,4,5);;
    static_assert(shape1.rank() == 5);
    static_assert(shape2[0] == 1);
    static_assert(shape1 == shape2); 
}

TEST(Test, ConstructorAndComparison) { 
    Strides explicit_strides(12, 4, 1);
    ASSERT_EQ(explicit_strides.rank(), 3);
    ASSERT_EQ(explicit_strides[0], 12);
    ASSERT_EQ(explicit_strides[1], 4);
    ASSERT_EQ(explicit_strides[2], 1);
 
    Shape shape(3, 4, 5);
    Strides computed_strides(shape);
 
    ASSERT_EQ(computed_strides.rank(), shape.rank());
    ASSERT_EQ(computed_strides[2], 1);
    ASSERT_EQ(computed_strides[1], 5);
    ASSERT_EQ(computed_strides[0], 20);
 
    Strides copy_strides(computed_strides);
    ASSERT_EQ(copy_strides, computed_strides);
    ASSERT_NE(explicit_strides, computed_strides);  // Different strides
 
    std::array<size_t, 3> arr = {7, 3, 1};
    Strides iter_strides(arr.begin(), arr.end());
    ASSERT_EQ(iter_strides.rank(), 3);
    ASSERT_EQ(iter_strides[0], 7);
    ASSERT_EQ(iter_strides[1], 3);
    ASSERT_EQ(iter_strides[2], 1);
}

TEST(Test, ConstexprStrides) { 

    constexpr Shape shape(2, 3, 4); 
    constexpr Strides strides(shape);
    static_assert((strides.rank() == 3)
        && (strides[0] == 12)
        && (strides[1] == 4)
        && (strides[2] == 1));
} 