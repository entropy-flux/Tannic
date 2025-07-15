#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "Tensor.hpp"
#include "Transformations.hpp"

using namespace tannic;

class MatmulTests : public ::testing::Test {
protected:
    void SetUp() override {}
};
 
TEST_F(MatmulTests, Basic) {
    Tensor X(float32, {3, 4}); X.initialize();
    Tensor Y(float32, {4, 5}); Y.initialize();
    
    float X_data[3 * 4] = {
        4.f, 2.f, 6.f, 0.f,
        1.f, 3.f, 5.f, 7.f,
        2.f, 4.f, 6.f, 8.f
    };
    
    float Y_data[4 * 5] = {
        1.f, 2.f, 3.f, 4.f, 5.f,
        0.f, 1.f, 0.f, 1.f, 0.f,
        2.f, 3.f, 4.f, 5.f, 6.f,
        1.f, 0.f, 1.f, 0.f, 1.f
    };
  
    float Z_expected[3 * 5] = {
        16.0f, 28.0f, 36.0f, 48.0f, 56.0f,
        18.0f, 20.0f, 30.0f, 32.0f, 42.0f,
        22.0f, 26.0f, 38.0f, 42.0f, 54.0f
    };
 
    float* x_ptr = reinterpret_cast<float*>(X.buffer()); for (int i = 0; i < 3*4; ++i) x_ptr[i] = X_data[i];
    float* y_ptr = reinterpret_cast<float*>(Y.buffer()); for (int i = 0; i < 4*5; ++i) y_ptr[i] = Y_data[i];

    Tensor Z = matmul(X, Y);   

 

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 3*5; ++i) { 
        EXPECT_NEAR(z_ptr[i], Z_expected[i], epsilon);
    }
}
 
TEST_F(MatmulTests, FirstTransposed) {
    Tensor X(float32, {2, 3}); X.initialize();
    Tensor Y(float32, {2, 3}); Y.initialize();
     
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    }; 

    float Y_data[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };
 
    float* x_ptr = reinterpret_cast<float*>(X.buffer()); for (int i = 0; i < 2*3; ++i) x_ptr[i] = X_data[i];
    float* y_ptr = reinterpret_cast<float*>(Y.buffer()); for (int i = 0; i < 2*3; ++i) y_ptr[i] = Y_data[i];
             
    Tensor Z = matmul(X.transpose(-1, -2), Y);

    float Z_expected[3*3] = {
        47.f, 52.f, 57.f,
        64.f, 71.f, 78.f,
        81.f, 90.f, 99.f
    };

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 3*3; ++i) { 
        EXPECT_NEAR(z_ptr[i], Z_expected[i], epsilon);
    }
}
 

TEST_F(MatmulTests, Batched) {
    Tensor A(float32, {2, 2, 2}); A.initialize();
    Tensor B(float32, {2, 2, 2}); B.initialize();
    
    float A_data[2][2][2] = {
        {{1.f, 2.f}, {3.f, 4.f}},
        {{5.f, 6.f}, {7.f, 8.f}}
    };

    float B_data[2][2][2] = {
        {{9.f, 8.f}, {7.f, 6.f}},
        {{5.f, 4.f}, {3.f, 2.f}}
    };

    // Fill tensors
    float* a_ptr = reinterpret_cast<float*>(A.buffer());
    float* b_ptr = reinterpret_cast<float*>(B.buffer());
    
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                a_ptr[i*4 + j*2 + k] = A_data[i][j][k];
            
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                b_ptr[i*4 + j*2 + k] = B_data[i][j][k];

    Tensor Z = matmul(A, B);
 
    float Z_expected[2][2][2] = {
        {
            {23.f, 20.f}, 
            {55.f, 48.f}
        },
        {
            {43.f, 32.f}, 
            {59.f, 44.f}
        }
    };

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                EXPECT_NEAR(z_ptr[i*4 + j*2 + k], Z_expected[i][j][k], epsilon)
                    << "Mismatch at batch " << i << ", (" << j << "," << k << ")";
            }
        }
    }
}
 
 
TEST_F(MatmulTests, SecondTransposed) {
    Tensor X(float32, {2, 3}); X.initialize();
    Tensor Y(float32, {2, 3}); Y.initialize();
    
    float X_data[2][3] = {
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    };

    float Y_data[2][3] = {
        {7.f, 8.f, 9.f},
        {10.f, 11.f, 12.f}
    };

    // Fill tensors
    float* x_ptr = reinterpret_cast<float*>(X.buffer());
    float* y_ptr = reinterpret_cast<float*>(Y.buffer());
    
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            x_ptr[i*3 + j] = X_data[i][j];
            
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            y_ptr[i*3 + j] = Y_data[i][j];
 

    Tensor Z = matmul(X, Y.transpose(-1, -2));

    float Z_expected[2][2] = {
        {50.f, 68.f},
        {122.f, 167.f}
    };

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(z_ptr[i*2 + j], Z_expected[i][j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}
 

TEST_F(MatmulTests, BothTransposed) {
    Tensor X(float32, {3, 2}); X.initialize();
    Tensor Y(float32, {2, 3}); Y.initialize();
    
    float X_data[3][2] = {
        {1.f, 4.f},
        {2.f, 5.f},
        {3.f, 6.f}
    };

    float Y_data[2][3] = {
        {7.f, 8.f, 9.f},
        {10.f, 11.f, 12.f}
    };

    // Fill tensors
    float* x_ptr = reinterpret_cast<float*>(X.buffer());
    float* y_ptr = reinterpret_cast<float*>(Y.buffer());
    
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            x_ptr[i*2 + j] = X_data[i][j];
            
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            y_ptr[i*3 + j] = Y_data[i][j]; 

    Tensor Z = matmul(X.transpose(-1, -2), Y.transpose(-1, -2));

    float Z_expected[2][2] = {
        {50.f, 68.f},
        {122.f, 167.f}
    };

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(z_ptr[i*2 + j], Z_expected[i][j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
} 
 
TEST_F(MatmulTests, Rank4_SecondTransposed) {
    Tensor X(float32, {2, 2, 2, 4}); X.initialize();
    Tensor Y(float32, {2, 2, 3, 4}); Y.initialize();
    
    float X_data[2][2][2][4] = {
        {
            {{1, 2, 3, 4}, {5, 6, 7, 8}},
            {{1, 2, 3, 4}, {5, 6, 7, 8}}
        },
        {
            {{1, 2, 3, 4}, {5, 6, 7, 8}},
            {{1, 2, 3, 4}, {5, 6, 7, 8}}
        }
    };

    float Y_data[2][2][3][4] = {
        {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9,10,11,12}
            },
            {
                {13,14,15,16},
                {17,18,19,20},
                {21,22,23,24}
            }
        },
        {
            {
                {1, 0, 1, 0},
                {0, 1, 0, 1},
                {1, 1, 1, 1}
            },
            {
                {2, 2, 2, 2},
                {3, 3, 3, 3},
                {4, 4, 4, 4}
            }
        }
    };

    // Fill tensors
    float* x_ptr = reinterpret_cast<float*>(X.buffer());
    float* y_ptr = reinterpret_cast<float*>(Y.buffer());
    
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 4; ++l)
                    x_ptr[i*16 + j*8 + k*4 + l] = X_data[i][j][k][l];
            
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 4; ++l)
                    y_ptr[i*24 + j*12 + k*4 + l] = Y_data[i][j][k][l];
 

    Tensor Z = matmul(X, Y.transpose(-1, -2));

    float Z_expected[2][2][2][3] = {
        {
            {{30.f, 70.f, 110.f}, {70.f, 174.f, 278.f}},
            {{150.f, 190.f, 230.f}, {382.f, 486.f, 590.f}}
        },
        {
            {{4.f, 6.f, 10.f}, {12.f, 14.f, 26.f}},
            {{20.f, 30.f, 40.f}, {52.f, 78.f, 104.f}}
        }
    };

    float* z_ptr = reinterpret_cast<float*>(Z.buffer());
    float epsilon = 1e-5f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 3; ++l) {
                    int idx = i*12 + j*6 + k*3 + l;
                    EXPECT_NEAR(z_ptr[idx], Z_expected[i][j][k][l], epsilon)
                        << "Mismatch at (" << i << "," << j << "," << k << "," << l << ")";
                }
            }
        }
    }
} 

/*


to run on jupyter or colab.
 
import unittest
import numpy as np

class TestTensorOps(unittest.TestCase):
   

    def test_matmul_basic(self):
        X = np.array([
            [4., 2., 6., 0.],
            [1., 3., 5., 7.],
            [2., 4., 6., 8.]
        ], dtype=np.float32)

        Y = np.array([
            [1., 2., 3., 4., 5.],
            [0., 1., 0., 1., 0.],
            [2., 3., 4., 5., 6.],
            [1., 0., 1., 0., 1.]
        ], dtype=np.float32)

        Z_expected = np.array([
            [16., 28., 36., 48., 56.],
            [18., 20., 30., 32., 42.],
            [22., 26., 38., 42., 54.]
        ], dtype=np.float32)

        Z = X @ Y
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)


    def test_matmul_first_transposed(self):
        X = np.array([
            [1., 2., 3.],
            [4., 5., 6.]
        ], dtype=np.float32)

        Y = np.array([
            [7., 8., 9.],
            [10., 11., 12.]
        ], dtype=np.float32)

        Z_expected = np.array([
            [47., 52., 57.],
            [64., 71., 78.],
            [81., 90., 99.]
        ], dtype=np.float32)

        Z = X.T @ Y
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)


    def test_matmul_second_transposed(self):
        X = np.array([
            [1., 2., 3.],
            [4., 5., 6.]
        ], dtype=np.float32)  # shape (2,3)

        Y = np.array([
            [7., 8., 9.],
            [10., 11., 12.]
        ], dtype=np.float32)  # shape (2,3)

        # Y.T shape (3,2)
        # X @ Y.T -> (2,3) @ (3,2) -> (2,2)
        Z_expected = np.array([
            [ 50.,  68.],
            [122., 167.]
        ], dtype=np.float32)

        Z = X @ Y.T
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)


    def test_matmul_both_transposed(self):
        X = np.array([
            [1., 4.],
            [2., 5.],
            [3., 6.]
        ], dtype=np.float32)

        Y = np.array([
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=np.float32)

        Z_expected = np.array([
            [50., 68.],
            [122., 167.]
        ], dtype=np.float32)

        Z = X.T @ Y
        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)



    def test_rank4_second_transposed(self):
        batch1, batch2 = 2, 2
        M, K, N = 2, 4, 3

        X = np.array([
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
            ],
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
            ]
        ], dtype=np.float32)

        Y = np.array([
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                ],
                [
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                ],
            ],
            [
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 1, 1, 1],
                ],
                [
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                ],
            ]
        ], dtype=np.float32)

        # Transpose last two dims of Y
        Y_t = np.transpose(Y, (0, 1, 3, 2))  # shape (2, 2, 4, 3)

        Z = np.zeros((batch1, batch2, M, N), dtype=np.float32)
        for b1 in range(batch1):
            for b2 in range(batch2):
                Z[b1, b2] = X[b1, b2] @ Y_t[b1, b2]

        Z_expected = np.array([
            [
                [30., 70., 110.],
                [70., 174., 278.]
            ],
            [
                [150., 190., 230.],
                [382., 486., 590.]
            ],
            [
                [4., 6., 10.],
                [12., 14., 26.]
            ],
            [
                [20., 30., 40.],
                [52., 78., 104.]
            ],
        ], dtype=np.float32).reshape(batch1, batch2, M, N)

        np.testing.assert_allclose(Z, Z_expected, rtol=1e-5, atol=1e-5)



if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


*/

 