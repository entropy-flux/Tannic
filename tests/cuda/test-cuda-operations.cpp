#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>  
#include "tensor.hpp" 
#include "comparisons.hpp"

using namespace tannic;

TEST(TestUnaryOpDVC, NegationDVC) {
    Tensor A(float32, {3, 3});
    A.initialize({
        {  9, 10, 11 },
        { 12, 13, 14 },
        { 15, 16, 17 }
    }, Device());

    Tensor expected(float32, {3, 3});
    expected.initialize({
        { -9, -10, -11 },
        { -12, -13, -14 },
        { -15, -16, -17 }
    }, Device());

    Tensor result = -A;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOpsDVC, AdditionDVC) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0., 1., 2.} }, { {3., 4., 5.} } }, Device());

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0., 10., 20.}, {30., 40., 50.}, {60., 70., 80.}, {90., 100., 110.} } }, Device());

    Tensor expected(float32, {2, 4, 3});
    expected.initialize({
        {
            {  0., 11., 22. },
            { 30., 41., 52. },
            { 60., 71., 82. },
            { 90.,101.,112. }
        },
        {
            {  3., 14., 25. },
            { 33., 44., 55. },
            { 63., 74., 85. },
            { 93.,104.,115. }
        }
    }, Device());

    Tensor result = A + B;
    std::cout << A.dtype() << B.dtype() << result.dtype() << std::endl;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOpsDVC, MultiplicationDVC) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } }, Device());

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } }, Device());

    Tensor expected(float32, {2, 4, 3});
    expected.initialize({
        {
            {   0,  10,  40 },
            {   0,  40, 100 },
            {   0,  70, 160 },
            {   0, 100, 220 }
        },
        {
            {   0,  40, 100 },
            {  90, 160, 250 },
            { 180, 280, 400 },
            { 270, 400, 550 }
        }
    }, Device());

    Tensor result = A * B;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOpsDVC, SubtractionDVC) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } }, Device());

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } }, Device());

    Tensor expected(float32, {2, 4, 3});
    expected.initialize({
        {
            {   0,  -9, -18 },
            { -30, -39, -48 },
            { -60, -69, -78 },
            { -90, -99,-108 }
        },
        {
            {   3,  -6, -15 },
            { -27, -36, -45 },
            { -57, -66, -75 },
            { -87, -96,-105 }
        }
    }, Device());

    Tensor result = A - B;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOpsDVC, ComplexDVC) {
    Tensor X(float32, {2,2});
    X.initialize({ {1, 6}, {2, 3} }, Device());
    X = complexify(X);

    Tensor Y(float32, {2,2});
    Y.initialize({ {2., 1.}, {1.5, 3.14} }, Device());
    Y = complexify(Y);

    Tensor result = realify(X * Y);

    Tensor expected(float32, {2,2});
    expected.initialize({
        { -4.00, 13.00 },
        { -6.42, 10.78 }
    }, Device());

    EXPECT_TRUE(allclose(result, expected, 1e-3f));
}

TEST(TestBinaryOpsDVC, PowerDVC) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } }, Device());

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } }, Device());

    Tensor result = A ^ B;
    ASSERT_EQ(result.shape(), Shape(2, 4, 3));

    Tensor expected(float32, {2, 4, 3});
    expected.initialize({
        {   
            { std::pow(0.f, 0.f),   std::pow(1.f, 10.f),   std::pow(2.f, 20.f) },
            { std::pow(0.f, 30.f),  std::pow(1.f, 40.f),   std::pow(2.f, 50.f) },
            { std::pow(0.f, 60.f),  std::pow(1.f, 70.f),   std::pow(2.f, 80.f) },
            { std::pow(0.f, 90.f),  std::pow(1.f, 100.f),  std::pow(2.f, 110.f) }
        },
        {   
            { std::pow(3.f, 0.f),   std::pow(4.f, 10.f),   std::pow(5.f, 20.f) },
            { std::pow(3.f, 30.f),  std::pow(4.f, 40.f),   std::pow(5.f, 50.f) },
            { std::pow(3.f, 60.f),  std::pow(4.f, 70.f),   std::pow(5.f, 80.f) },
            { std::pow(3.f, 90.f),  std::pow(4.f, 100.f),  std::pow(5.f, 110.f) }
        }
    }, Device());

    EXPECT_TRUE(allclose(result, expected, 0.01));
}

TEST(TestBinaryOpsDVC, BroadcastMultiplyAddDVC) {
    Tensor X(float32, {2, 3});
    X.initialize({
        { -1.22474f, 0.f, 1.22474f },
        { -1.22474f, 0.f, 1.22474f }
    }, Device());

    Tensor W(float32, {3});
    W.initialize({ 0.5f, 1.0f, 1.5f }, Device());

    Tensor b(float32, {3});
    b.initialize({ 0.0f, 0.1f, 0.2f }, Device());

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { -0.61237f, 0.1f, 2.03711f },
        { -0.61237f, 0.1f, 2.03711f }
    }, Device());

    Tensor result = X * W + b;
    EXPECT_TRUE(allclose(result, expected, 1e-5f));
}

#endif