#include <gtest/gtest.h>
#include "Tensor.hpp"

TEST(Test, Constructors) {
    Tensor a(Shape(1,2,3,4), float32);
    ASSERT_EQ(a.dtype(), float32);
    Tensor b(a.shape(), float32);
    ASSERT_EQ(a.shape(), b.shape());
    Tensor c({1,2,3,4}, float32);
    ASSERT_EQ(a.shape(), c.shape());
    ASSERT_EQ(b.shape(), c.shape());
    ASSERT_EQ(a.dtype(), c.dtype());
    ASSERT_EQ(b.dtype(), c.dtype());
}