#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "tensor.hpp"
#include "functions.hpp"
#include "tannic/comparisons.hpp"
 
using namespace tannic;

TEST(FunctionTests, Log) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::log(1.0f), std::log(2.0f), std::log(3.0f) },
        { std::log(4.0f), std::log(5.0f), std::log(6.0f) }
    });

    Tensor result = log(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Exp) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::exp(1.0f), std::exp(2.0f), std::exp(3.0f) },
        { std::exp(4.0f), std::exp(5.0f), std::exp(6.0f) }
    });

    Tensor result = exp(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Sqrt) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f) },
        { std::sqrt(4.0f), std::sqrt(5.0f), std::sqrt(6.0f) }
    });

    Tensor result = sqrt(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Rsqrt) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    float eps = 1e-6f;
    Tensor expected(float32, {2, 3});
    expected.initialize({
        { 1.0f / std::sqrt(1.0f + eps), 1.0f / std::sqrt(2.0f + eps), 1.0f / std::sqrt(3.0f + eps) },
        { 1.0f / std::sqrt(4.0f + eps), 1.0f / std::sqrt(5.0f + eps), 1.0f / std::sqrt(6.0f + eps) }
    });

    Tensor result = rsqrt(A, eps);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Abs) {
    Tensor A(float32, {2, 3});
    A.initialize({ {-1, -2, -3}, {-4, -5, -6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor result = abs(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Sin) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::sin(1.0f), std::sin(2.0f), std::sin(3.0f) },
        { std::sin(4.0f), std::sin(5.0f), std::sin(6.0f) }
    });

    Tensor result = sin(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Cos) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::cos(1.0f), std::cos(2.0f), std::cos(3.0f) },
        { std::cos(4.0f), std::cos(5.0f), std::cos(6.0f) }
    });

    Tensor result = cos(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Tan) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::tan(1.0f), std::tan(2.0f), std::tan(3.0f) },
        { std::tan(4.0f), std::tan(5.0f), std::tan(6.0f) }
    });

    Tensor result = tan(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Sinh) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::sinh(1.0f), std::sinh(2.0f), std::sinh(3.0f) },
        { std::sinh(4.0f), std::sinh(5.0f), std::sinh(6.0f) }
    });

    Tensor result = sinh(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Cosh) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::cosh(1.0f), std::cosh(2.0f), std::cosh(3.0f) },
        { std::cosh(4.0f), std::cosh(5.0f), std::cosh(6.0f) }
    });

    Tensor result = cosh(A);
    EXPECT_TRUE(allclose(result, expected));
}

TEST(FunctionTests, Tanh) {
    Tensor A(float32, {2, 3});
    A.initialize({ {1, 2, 3}, {4, 5, 6} });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { std::tanh(1.0f), std::tanh(2.0f), std::tanh(3.0f) },
        { std::tanh(4.0f), std::tanh(5.0f), std::tanh(6.0f) }
    });

    Tensor result = tanh(A);
    EXPECT_TRUE(allclose(result, expected));
}
