#include <gtest/gtest.h>
#include "Parameter.hpp"

TEST(Test, Constructors) {
    constexpr Parameter a(Shape{1,2,3,4}, float32);
    static_assert(a.dtype() == float32);
    constexpr Parameter b(a.shape(), float32);
    static_assert(a.shape() == b.shape());
    Parameter c(Shape{1,2,3,4}, float32);
    ASSERT_EQ(a.shape(), c.shape());
    ASSERT_EQ(b.shape(), c.shape());
    ASSERT_EQ(a.dtype(), c.dtype());
    ASSERT_EQ(b.dtype(), c.dtype());
}