#include <gtest/gtest.h>
#include "Tensor.hpp"

TEST(Test, Indexing) { 
    Tensor x({1,2,3}, float32);
    x[0][0][0] = 1.f;
    x[0][0][1] = 2.f;
    x[0][0][2] = 3.f;
    x[0][1][0] = 4.f;
    x[0][1][1] = 5.f;
    x[0][1][2] = 6.f;

    Tensor y = x[0];
    ASSERT_EQ(y[0][0], 1.f);
    ASSERT_EQ(y[0][1], 2.f);
    ASSERT_EQ(y[0][2], 3.f);
    ASSERT_EQ(y[1][0], 4.f);
    ASSERT_EQ(y[1][1], 5.f);
    ASSERT_EQ(y[1][2], 6.f);

    Tensor z = y[1];
    ASSERT_EQ(z[0], 4.f);
    ASSERT_EQ(z[1], 5.f);
    ASSERT_EQ(z[2], 6.f);

    Tensor w = z[1];
    float w_value = w.item<float>();
    ASSERT_EQ(w_value, 5.f);
}