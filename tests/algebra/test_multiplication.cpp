#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Operations.hpp"


TEST(Test, MatrixMulFloat) {
    Tensor a(float32, {2, 3});
    Tensor b(float32, {2, 3});

    a[0][0] = 1.f;  a[0][1] = 2.f;  a[0][2] = 3.f;
    a[1][0] = 4.f;  a[1][1] = 5.f;  a[1][2] = 6.f;

    b[0][0] = 6.f;  b[0][1] = -5.f; b[0][2] = 4.f;
    b[1][0] = 3.f;  b[1][1] = 2.f;  b[1][2] = 1.f;

    Tensor c = a * b;

    EXPECT_NEAR(c[0][0].item<float>(), 6.f, 1e-5);
    EXPECT_NEAR(c[0][1].item<float>(), -10.f, 1e-5);
    EXPECT_NEAR(c[0][2].item<float>(), 12.f, 1e-5);

    EXPECT_NEAR(c[1][0].item<float>(), 12.f, 1e-5);
    EXPECT_NEAR(c[1][1].item<float>(), 10.f, 1e-5);
    EXPECT_NEAR(c[1][2].item<float>(), 6.f, 1e-5);
}

#ifndef OPENBLAS
TEST(Test, MatrixMulFloatDouble) {
    Tensor a(float32, {2, 3});
    Tensor b(float64, {2, 3});

    a[0][0] = 1.f;  a[0][1] = 2.f;  a[0][2] = 3.f;
    a[1][0] = 4.f;  a[1][1] = 5.f;  a[1][2] = 6.f;

    b[0][0] = 6.0;  b[0][1] = -5.0; b[0][2] = 4.0;
    b[1][0] = 3.0;  b[1][1] = 2.0;  b[1][2] = 1.0;

    Tensor c = a * b;

    EXPECT_NEAR(c[0][0].item<double>(), 6.0, 1e-5);
    EXPECT_NEAR(c[0][1].item<double>(), -10.0, 1e-5);
    EXPECT_NEAR(c[0][2].item<double>(), 12.0, 1e-5);

    EXPECT_NEAR(c[1][0].item<double>(), 12.0, 1e-5);
    EXPECT_NEAR(c[1][1].item<double>(), 10.0, 1e-5);
    EXPECT_NEAR(c[1][2].item<double>(), 6.0, 1e-5);
} 
#endif