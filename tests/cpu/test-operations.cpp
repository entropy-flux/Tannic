#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include "tensor.hpp"   
#include "comparisons.hpp"

using namespace tannic;  


TEST(TestUnaryOp, Negation) {
    Tensor A(float32, {3, 3});
    A.initialize({
        {  9, 10, 11 },
        { 12, 13, 14 },
        { 15, 16, 17 }
    });

    Tensor expected(float32, {3, 3});
    expected.initialize({
        { -9, -10, -11 },
        { -12, -13, -14 },
        { -15, -16, -17 }
    });

    Tensor result = -A;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOps, Addition) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0., 1., 2.} }, { {3., 4., 5.} } });

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0., 10., 20.}, {30., 40., 50.}, {60., 70., 80.}, {90., 100., 110.} } });

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
    });

    Tensor result = A + B;
    std::cout << A.dtype() << B.dtype() << result.dtype() << std::endl;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOps, Multiplication) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } });

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } });

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
    });

    Tensor result = A * B;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOps, Subtraction) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } });

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } });

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
    });

    Tensor result = A - B;
    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOps, Complex) {
    Tensor X(float32, {2,2});
    X.initialize({ {1, 6}, {2, 3} });
    X = complexify(X);

    Tensor Y(float32, {2,2});
    Y.initialize({ {2., 1.}, {1.5, 3.14} });
    Y = complexify(Y);

    Tensor result = realify(X * Y);

    Tensor expected(float32, {2,2});
    expected.initialize({
        { -4.00, 13.00 },
        { -6.42, 10.78 }
    });

    EXPECT_TRUE(allclose(result, expected, 1e-3f));
}

TEST(TestBinaryOps, Power) {
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0, 1, 2} }, { {3, 4, 5} } });

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0, 10, 20}, {30, 40, 50}, {60, 70, 80}, {90, 100, 110} } });

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
    });

    EXPECT_TRUE(allclose(result, expected));
}

TEST(TestBinaryOps, BroadcastMultiplyAdd) {
    Tensor X(float32, {2, 3});
    X.initialize({
        { -1.22474f, 0.f, 1.22474f },
        { -1.22474f, 0.f, 1.22474f }
    });

    Tensor W(float32, {3});
    W.initialize({ 0.5f, 1.0f, 1.5f });

    Tensor b(float32, {3});
    b.initialize({ 0.0f, 0.1f, 0.2f });

    Tensor expected(float32, {2, 3});
    expected.initialize({
        { -0.61237f, 0.1f, 2.03711f },
        { -0.61237f, 0.1f, 2.03711f }
    });

    Tensor result = X * W + b;
    EXPECT_TRUE(allclose(result, expected, 1e-5f));
}





/* 

to run on jupyter or colab.

import unittest
import numpy as np

class TestTensorOps(unittest.TestCase):

    def setUp(self):
        # Shapes
        self.shape_A = (2, 1, 3)
        self.shape_B = (1, 4, 3) 

        # Data initialization matching your C++ arrays
        self.A = np.arange(6, dtype=np.float32).reshape(self.shape_A)
        self.B = (np.arange(12, dtype=np.float32) * 10).reshape(self.shape_B) 

    def test_addition(self):
        self.C = self.A + self.B
        expected = np.array([
            [
                [  0.0,  11.0,  22.0],
                [ 30.0,  41.0,  52.0],
                [ 60.0,  71.0,  82.0],
                [ 90.0, 101.0, 112.0]
            ],
            [
                [  3.0,  14.0,  25.0],
                [ 33.0,  44.0,  55.0],
                [ 63.0,  74.0,  85.0],
                [ 93.0, 104.0, 115.0]
            ]
        ], dtype=np.float32)
        np.testing.assert_allclose(self.C, expected, rtol=1e-6)

    def test_multiplication(self):
        self.C = self.A * self.B
        expected = np.array([
            [
                [  0.0,  10.0,  40.0],
                [  0.0,  40.0, 100.0],
                [  0.0,  70.0, 160.0],
                [  0.0, 100.0, 220.0]
            ],
            [
                [  0.0,  40.0, 100.0],
                [ 90.0, 160.0, 250.0],
                [180.0, 280.0, 400.0],
                [270.0, 400.0, 550.0]
            ]
        ], dtype=np.float32)
        np.testing.assert_allclose(self.C, expected, rtol=1e-6)

    def test_subtraction(self):
        self.C = self.A - self.B
        expected = np.array([
            [
                [  0.0,  -9.0, -18.0],
                [-30.0, -39.0, -48.0],
                [-60.0, -69.0, -78.0],
                [-90.0, -99.0, -108.0]
            ],
            [
                [  3.0,  -6.0, -15.0],
                [-27.0, -36.0, -45.0],
                [-57.0, -66.0, -75.0],
                [-87.0, -96.0, -105.0]
            ]
        ], dtype=np.float32)
        np.testing.assert_allclose(self.C, expected, rtol=1e-6)

    def test_negation(self):
        self.A = -self.A
        expected = np.array([
            [
                [-0.0, -1.0, -2.0]
            ],
            [
                [-3.0, -4.0, -5.0]
            ]
        ], dtype=np.float32)
        np.testing.assert_allclose(self.A, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
*/
 