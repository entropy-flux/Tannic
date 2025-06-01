#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Algebra/Operations.hpp"

TEST(Test, MatrixSubFloat) {
    Tensor a({2, 3}, float32);
    Tensor b({2, 3}, float32);

    a[0][0] = 7.f;  a[0][1] = 7.f;  a[0][2] = 7.f;
    a[1][0] = 7.f;  a[1][1] = 7.f;  a[1][2] = 7.f;

    b[0][0] = 6.f;  b[0][1] = -5.f; b[0][2] = 4.f;
    b[1][0] = 3.f;  b[1][1] = 2.f;  b[1][2] = 1.f;

    Tensor c = a - b;

    EXPECT_NEAR(c[0][0].item<float>(), 1.f, 1e-5);
    EXPECT_NEAR(c[0][1].item<float>(), 12.f, 1e-5);
    EXPECT_NEAR(c[0][2].item<float>(), 3.f, 1e-5);

    EXPECT_NEAR(c[1][0].item<float>(), 4.f, 1e-5);
    EXPECT_NEAR(c[1][1].item<float>(), 5.f, 1e-5);
    EXPECT_NEAR(c[1][2].item<float>(), 6.f, 1e-5);
}

TEST(Test, MatrixSubFloatDouble) {
    Tensor a({2, 3}, float64);
    Tensor b({2, 3}, float32);

    a[0][0] = 7.0;  a[0][1] = 7.0;  a[0][2] = 7.0;
    a[1][0] = 7.0;  a[1][1] = 7.0;  a[1][2] = 7.0;

    b[0][0] = 6.f;  b[0][1] = -5.f; b[0][2] = 4.f;
    b[1][0] = 3.f;  b[1][1] = 2.f;  b[1][2] = 1.f;

    Tensor c = a - b;

    EXPECT_NEAR(c[0][0].item<double>(), 1.0, 1e-5);
    EXPECT_NEAR(c[0][1].item<double>(), 12.0, 1e-5);
    EXPECT_NEAR(c[0][2].item<double>(), 3.0, 1e-5);

    EXPECT_NEAR(c[1][0].item<double>(), 4.0, 1e-5);
    EXPECT_NEAR(c[1][1].item<double>(), 5.0, 1e-5);
    EXPECT_NEAR(c[1][2].item<double>(), 6.0, 1e-5);
}
