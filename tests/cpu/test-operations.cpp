#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
 
#include "Tensor.hpp"   

using namespace tannic; 

TEST(TestUnaryOp, Negation) {
    Tensor A(float32, {3, 3}); A.initialize(); 
    float* A_data = reinterpret_cast<float*>(A.bytes());
    for(int i = 0; i < 3*3; i++) {
        A_data[i] = 9 + 1. * i;
    }

    Tensor B = -A;
    float* B_data = reinterpret_cast<float*>(B.bytes());

    float expected[] = {
        -9.0, -10.0, -11.0,                
        -12.0, -13.0, -14.0,
        -15.0, -16.0, -17.0
    };
    
    for(int i = 0; i < 3*3; i++) {
        ASSERT_FLOAT_EQ(B_data[i], expected[i]);
    }
}


class TestBinaryOps : public ::testing::Test {
protected:
    Tensor A; 
    Tensor B; 

    TestBinaryOps() 
    :   A(float32, Shape(2, 1, 3))
    ,   B(float32, Shape(1, 4, 3))
    { 
        A.initialize();  
        B.initialize();  
    }

    void SetUp() override {    
        float* A_data = reinterpret_cast<float*>(A.bytes());
        float* B_data = reinterpret_cast<float*>(B.bytes());

        const float A_init[2][1][3] = {
            {{0.0f, 1.0f, 2.0f}}, 
            {{3.0f, 4.0f, 5.0f}} 
        };
        
        const float B_init[1][4][3] = {
            {
                {0.0f,  10.0f, 20.0f},  
                {30.0f, 40.0f, 50.0f},  
                {60.0f, 70.0f, 80.0f}, 
                {90.0f, 100.0f, 110.0f}
            }
        };

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 1; ++j) {
                for (int k = 0; k < 3; ++k) {
                    A_data[i * 3 + j * 3 + k] = A_init[i][j][k];
                }
            }
        }

        for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 3; ++k) {
                    B_data[j * 3 + k] = B_init[i][j][k]; 
                }
            }
        }
    }
}; 

TEST_F(TestBinaryOps, Addition) {
    Tensor C = A + B;
    ASSERT_EQ(C.shape(), Shape(2, 4, 3));
    float expected[2][4][3] = {
        {
            {  0.0f,  11.0f,  22.0f},
            { 30.0f,  41.0f,  52.0f},
            { 60.0f,  71.0f,  82.0f}, 
            { 90.0f, 101.0f, 112.0f}
        },
        {
            {  3.0f,  14.0f,  25.0f},
            { 33.0f,  44.0f,  55.0f},
            { 63.0f,  74.0f,  85.0f},
            { 93.0f, 104.0f, 115.0f}
        }
    };

    float* C_data = reinterpret_cast<float*>(C.bytes());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) 
                    << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestBinaryOps, Multiplication) { 
    Tensor C = A * B;
    float expected[2][4][3] = {
        {
            {  0.0f,  10.0f,  40.0f},
            {  0.0f,  40.0f,  100.0f},
            {  0.0f,  70.0f,  160.0f},
            {  0.0f, 100.0f,  220.0f}
        },
        {
            {  0.0f, 40.0f,  100.0f},
            { 90.0f, 160.0f,  250.0f},
            { 180.0f, 280.0f,  400.0f},
            { 270.0f, 400.0f,  550.0f}
        }
    };

    float* C_data = reinterpret_cast<float*>(C.bytes());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) 
                    << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestBinaryOps, Subtraction) {
    Tensor C = A - B; 
    float expected[2][4][3] = {
        {
            {  0.0f,  -9.0f, -18.0f },
            { -30.0f, -39.0f, -48.0f },
            { -60.0f, -69.0f, -78.0f },
            { -90.0f, -99.0f, -108.0f }
        },
        {
            {  3.0f,  -6.0f, -15.0f },
            { -27.0f, -36.0f, -45.0f },
            { -57.0f, -66.0f, -75.0f },
            { -87.0f, -96.0f, -105.0f }
        }
    };

    float* C_data = reinterpret_cast<float*>(C.bytes());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) 
                    << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
} 

TEST_F(TestBinaryOps, Complex) {
    Tensor X(float32, {2,2}); X.initialize();  
    X[0, 0] = 1;
    X[0, 1] = 6;
    X[1, 0] = 2;
    X[1, 1] = 3;      
    X = complex(X);  
    
    Tensor Y(float32, {2,2}); Y.initialize(); 
    Y[0, 0] = 2;
    Y[0, 1] = 1;
    Y[1, 0] = 1.5;
    Y[1, 1] = 3.14;   
    Y = complex(Y);
 

    Tensor Z = real(X*Y); 
 
    float* Z_data = reinterpret_cast<float*>(Z.bytes());
    ASSERT_NEAR(Z_data[0], -4.00, 0.001);
    ASSERT_NEAR(Z_data[1], 13.00, 0.001);
    ASSERT_NEAR(Z_data[2], -6.42, 0.001);
    ASSERT_NEAR(Z_data[3], 10.78, 0.001);
}

TEST_F(TestBinaryOps, Power) { 
    Tensor C = A ^ B;
    ASSERT_EQ(C.shape(), Shape(2, 4, 3));

    float expected[2][4][3];
    float* A_data = reinterpret_cast<float*>(A.bytes());
    float* B_data = reinterpret_cast<float*>(B.bytes());
 
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                int idx_A = i * 3 + k;       
                int idx_B = j * 3 + k; 
                expected[i][j][k] = std::pow(A_data[idx_A], B_data[idx_B]);
            }
        }
    }
 
    float* C_data = reinterpret_cast<float*>(C.bytes());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k])
                    << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
        }
    }
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
 