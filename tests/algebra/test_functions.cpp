#include <gtest/gtest.h>
#include <cmath>
#include "Tensor.hpp"
#include "Operations.hpp"
#include "Functions.hpp"

constexpr float EPS_F = 1e-5f;
constexpr double EPS_D = 1e-12;


TEST(TestExpr, LogFloat32) {
    Tensor a(float32, {3});
    a[0] = 1.f; a[1] = std::exp(1.f); a[2] = 100.f;

    Tensor out = log(a);

    EXPECT_NEAR(out[0].item<float>(), 0.f, EPS_F);
    EXPECT_NEAR(out[1].item<float>(), 1.f, EPS_F);
    EXPECT_NEAR(out[2].item<float>(), std::log(100.f), EPS_F);
}

TEST(TestExpr, ExpFloat64) {
    Tensor a(float64, {3});
    a[0] = 0.0; a[1] = 1.0; a[2] = 2.0;

    Tensor out = exp(a);

    EXPECT_NEAR(out[0].item<double>(), 1.0, EPS_D);
    EXPECT_NEAR(out[1].item<double>(), std::exp(1.0), EPS_D);
    EXPECT_NEAR(out[2].item<double>(), std::exp(2.0), EPS_D);
}

TEST(TestExpr, SqrtFloat32) {
    Tensor a(float32, {3});
    a[0] = 0.f; a[1] = 4.f; a[2] = 9.f;

    Tensor out = sqrt(a);

    EXPECT_NEAR(out[0].item<float>(), 0.f, EPS_F);
    EXPECT_NEAR(out[1].item<float>(), 2.f, EPS_F);
    EXPECT_NEAR(out[2].item<float>(), 3.f, EPS_F);
}

TEST(TestExpr, AbsFloat64) {
    Tensor a(float64, {3});
    a[0] = -2.0; a[1] = 0.0; a[2] = 7.0;

    Tensor out = abs(a);

    EXPECT_NEAR(out[0].item<double>(), 2.0, EPS_D);
    EXPECT_NEAR(out[1].item<double>(), 0.0, EPS_D);
    EXPECT_NEAR(out[2].item<double>(), 7.0, EPS_D);
}

TEST(TestExpr, SinFloat32) {
    Tensor a(float32, {3});
    a[0] = 0.f; a[1] = static_cast<float>(M_PI / 2); a[2] = static_cast<float>(M_PI);

    Tensor out = sin(a);

    EXPECT_NEAR(out[0].item<float>(), 0.f, EPS_F);
    EXPECT_NEAR(out[1].item<float>(), 1.f, EPS_F);
    EXPECT_NEAR(out[2].item<float>(), 0.f, EPS_F);
}

TEST(TestExpr, SinhFloat64) {
    Tensor a(float64, {3});
    a[0] = 0.0; a[1] = 1.0; a[2] = 2.0;

    Tensor out = sinh(a);

    EXPECT_NEAR(out[0].item<double>(), 0.0, EPS_D);
    EXPECT_NEAR(out[1].item<double>(), std::sinh(1.0), EPS_D);
    EXPECT_NEAR(out[2].item<double>(), std::sinh(2.0), EPS_D);
}

TEST(TestExpr, CosFloat32) {
    Tensor a(float32, {3});
    a[0] = 0.f; a[1] = static_cast<float>(M_PI / 2); a[2] = static_cast<float>(M_PI);

    Tensor out = cos(a);

    EXPECT_NEAR(out[0].item<float>(), 1.f, EPS_F);
    EXPECT_NEAR(out[1].item<float>(), 0.f, EPS_F);
    EXPECT_NEAR(out[2].item<float>(), -1.f, EPS_F);
}

TEST(TestExpr, CoshFloat64) {
    Tensor a(float64, {3});
    a[0] = 0.0; a[1] = 1.0; a[2] = 2.0;

    Tensor out = cosh(a);

    EXPECT_NEAR(out[0].item<double>(), 1.0, EPS_D);
    EXPECT_NEAR(out[1].item<double>(), std::cosh(1.0), EPS_D);
    EXPECT_NEAR(out[2].item<double>(), std::cosh(2.0), EPS_D);
}

TEST(TestExpr, TanFloat32) {
    Tensor a(float32, {3});
    a[0] = 0.f; a[1] = static_cast<float>(M_PI / 4); a[2] = static_cast<float>(M_PI / 3);

    Tensor out = tan(a);

    EXPECT_NEAR(out[0].item<float>(), 0.f, EPS_F);
    EXPECT_NEAR(out[1].item<float>(), 1.f, EPS_F);
    EXPECT_NEAR(out[2].item<float>(), std::tan(M_PI / 3), EPS_F);
}

TEST(TestExpr, TanhFloat64) {
    Tensor a(float64, {3});
    a[0] = 0.0; a[1] = 1.0; a[2] = 2.0;

    Tensor out = tanh(a);

    EXPECT_NEAR(out[0].item<double>(), 0.0, EPS_D);
    EXPECT_NEAR(out[1].item<double>(), std::tanh(1.0), EPS_D);
    EXPECT_NEAR(out[2].item<double>(), std::tanh(2.0), EPS_D);
}
