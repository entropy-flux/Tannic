#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Operations.hpp"


TEST(Test, NegateFloat) {
    Tensor a(float32, {2, 3});

    a[0][0] = 1.f;  a[0][1] = -2.f; a[0][2] = 3.f;
    a[1][0] = -4.f; a[1][1] = 5.f;  a[1][2] = -6.f;

    Tensor b = -a;

    EXPECT_NEAR(b[0][0].item<float>(), -1.f, 1e-5);
    EXPECT_NEAR(b[0][1].item<float>(), 2.f, 1e-5);
    EXPECT_NEAR(b[0][2].item<float>(), -3.f, 1e-5);

    EXPECT_NEAR(b[1][0].item<float>(), 4.f, 1e-5);
    EXPECT_NEAR(b[1][1].item<float>(), -5.f, 1e-5);
    EXPECT_NEAR(b[1][2].item<float>(), 6.f, 1e-5);
}

TEST(Test, NegateDouble) {
    Tensor a(float64, {2, 3});

    a[0][0] = 1.0;  a[0][1] = -2.0; a[0][2] = 3.0;
    a[1][0] = -4.0; a[1][1] = 5.0;  a[1][2] = -6.0;

    Tensor b = -a;

    EXPECT_NEAR(b[0][0].item<double>(), -1.0, 1e-5);
    EXPECT_NEAR(b[0][1].item<double>(), 2.0, 1e-5);
    EXPECT_NEAR(b[0][2].item<double>(), -3.0, 1e-5);

    EXPECT_NEAR(b[1][0].item<double>(), 4.0, 1e-5);
    EXPECT_NEAR(b[1][1].item<double>(), -5.0, 1e-5);
    EXPECT_NEAR(b[1][2].item<double>(), 6.0, 1e-5);
}
