#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Algebra/Operations.hpp"

TEST(Test, MatrixAddFloat) {
    Tensor a({2, 3}, float32);
    Tensor b({2, 3}, float32);

    a[0][0] = 1.f; a[0][1] = 12.f; a[0][2] = 3.f;
    a[1][0] = 4.f; a[1][1] = 5.f; a[1][2] = 6.f;

    b[0][0] = 6.f; b[0][1] = -5.f; b[0][2] = 4.f;
    b[1][0] = 3.f; b[1][1] = 2.f; b[1][2] = 1.f;

    Tensor c = a + b;

    EXPECT_NEAR(c[0][0].item<float>(), 7.f, 1e-5);
    EXPECT_NEAR(c[0][1].item<float>(), 7.f, 1e-5);
    EXPECT_NEAR(c[0][2].item<float>(), 7.f, 1e-5);

    EXPECT_NEAR(c[1][0].item<float>(), 7.f, 1e-5);
    EXPECT_NEAR(c[1][1].item<float>(), 7.f, 1e-5);
    EXPECT_NEAR(c[1][2].item<float>(), 7.f, 1e-5);
}


#ifndef OPENBLAS
TEST(Test, MatrixAddFloatDouble) {
    Tensor a({2, 3}, float32);
    Tensor b({2, 3}, float64);

    a[0][0] = 1.f; a[0][1] = 12.f; a[0][2] = 3.f;
    a[1][0] = 4.f; a[1][1] = 5.f; a[1][2] = 6.f;

    b[0][0] = 6.0; b[0][1] = -5.0; b[0][2] = 4.0;
    b[1][0] = 3.0; b[1][1] = 2.0; b[1][2] = 1.0;

    Tensor c = a + b;

    EXPECT_NEAR(c[0][0].item<double>(), 7.0, 1e-5);
    EXPECT_NEAR(c[0][1].item<double>(), 7.0, 1e-5);
    EXPECT_NEAR(c[0][2].item<double>(), 7.0, 1e-5);

    EXPECT_NEAR(c[1][0].item<double>(), 7.0, 1e-5);
    EXPECT_NEAR(c[1][1].item<double>(), 7.0, 1e-5);
    EXPECT_NEAR(c[1][2].item<double>(), 7.0, 1e-5);
} 
#endif