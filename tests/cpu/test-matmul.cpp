#include <gtest/gtest.h>
#include <cstring>
#include <cmath>

#include "core/types.h"
#include "core/tensor.h" 
#include "cpu/matmul-op.hpp"  

TEST(MatmulTests, Basic) {
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

    float Z_data[3 * 5] = {0};

    float Z_expected_data[3 * 5] = {
        16.0f, 28.0f, 36.0f, 48.0f, 56.0f,
        18.0f, 20.0f, 30.0f, 32.0f, 42.0f,
        22.0f, 26.0f, 38.0f, 42.0f, 54.0f
    };

    size_t shape_X[2] = {3, 4};
    size_t shape_Y[2] = {4, 5};
    size_t shape_Z[2] = {3, 5};

    size_t strides_X[2] = {4, 1};
    size_t strides_Y[2] = {5, 1};
    size_t strides_Z[2] = {5, 1};

    tensor_t X = {
        .rank = 2, 
        .shape = shape_X, 
        .strides = strides_X, 
        .storage = {.address = X_data, .nbytes = sizeof(X_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Y = {
        .rank = 2, 
        .shape = shape_Y, 
        .strides = strides_Y, 
        .storage = {.address = Y_data, .nbytes = sizeof(Y_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 2, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data)); 

    cpu::matmul::kernels[cpu::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, false);

    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(((float*)Z.storage.address)[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(MatmulTests, FirstTransposed) {
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    float Y_data_fixed[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };

    float Z_data[3 * 3] = {0};

    size_t shape_X[2] = {2, 3};  
    size_t shape_Y[2] = {2, 3}; 
    size_t shape_Z[2] = {3, 3};  

    size_t strides_X[2] = {3, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {3, 1};

    tensor_t X = {
        .rank = 2, 
        .shape = shape_X, 
        .strides = strides_X, 
        .storage = {.address = X_data, .nbytes = sizeof(X_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Y = {
        .rank = 2, 
        .shape = shape_Y, 
        .strides = strides_Y, 
        .storage = {.address = Y_data_fixed, .nbytes = sizeof(Y_data_fixed), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 2, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data));
 
    cpu::matmul::kernels[cpu::index(X.dtype,Y.dtype)](&X, &Y, &Z, true, false);

    float Z_expected_data[3 * 3] = {
        47.f, 52.f, 57.f,
        64.f, 71.f, 78.f,
        81.f, 90.f, 99.f
    };

    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(((float*)Z.storage.address)[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(MatmulTests, SecondTransposed) {
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    float Y_data[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };

    float Z_data[2 * 2] = {0};

    size_t shape_X[2] = {2, 3};
    size_t shape_Y[2] = {2, 3};
    size_t shape_Z[2] = {2, 2};

    size_t strides_X[2] = {3, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {2, 1};

    tensor_t X = {
        .rank = 2, 
        .shape = shape_X, 
        .strides = strides_X, 
        .storage = {.address = X_data, .nbytes = sizeof(X_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Y = {
        .rank = 2, 
        .shape = shape_Y, 
        .strides = strides_Y, 
        .storage = {.address = Y_data, .nbytes = sizeof(Y_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 2, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data));

    cpu::matmul::kernels[cpu::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, true);

    float Z_expected_data[2 * 2] = {
        50.f, 68.f,
        122.f, 167.f
    };

    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(((float*)Z.storage.address)[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(MatmulTests, BothTransposed) {
    float X_data[3 * 2] = {
        1.f, 4.f,
        2.f, 5.f,
        3.f, 6.f
    };

    float Y_data[3 * 2] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f 
    };

    float Z_data[2 * 2] = {0};

    size_t shape_X[2] = {3, 2};
    size_t shape_Y[2] = {2, 3};
    size_t shape_Z[2] = {2, 2};

    size_t strides_X[2] = {2, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {2, 1};

    tensor_t X = {
        .rank = 2, 
        .shape = shape_X, 
        .strides = strides_X, 
        .storage = {.address = X_data, .nbytes = sizeof(X_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Y = {
        .rank = 2, 
        .shape = shape_Y, 
        .strides = strides_Y, 
        .storage = {.address = Y_data, .nbytes = sizeof(Y_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 2, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data));
 
    cpu::matmul::kernels[cpu::index(X.dtype,Y.dtype)](&X, &Y, &Z, true, true);

    float Z_expected_data[2 * 2] = {
        50.f, 68.f,
        122.f, 167.f
    };

    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(((float*)Z.storage.address)[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(MatmulTests, Batched) {
    float A_data[2 * 2 * 2] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f,
        7.f, 8.f
    };

    float B_data[2 * 2 * 2] = {
        9.f, 8.f,
        7.f, 6.f,
        5.f, 4.f,
        3.f, 2.f
    };

    float Z_data[2 * 2 * 2] = {0};

    size_t shape_A[3] = {2, 2, 2};
    size_t shape_B[3] = {2, 2, 2};
    size_t shape_Z[3] = {2, 2, 2};

    size_t strides_A[3] = {4, 2, 1};
    size_t strides_B[3] = {4, 2, 1};
    size_t strides_Z[3] = {4, 2, 1};

    tensor_t A = {
        .rank = 3, 
        .shape = shape_A, 
        .strides = strides_A, 
        .storage = {.address = A_data, .nbytes = sizeof(A_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t B = {
        .rank = 3, 
        .shape = shape_B, 
        .strides = strides_B, 
        .storage = {.address = B_data, .nbytes = sizeof(B_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 3, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data));
 
    cpu::matmul::kernels[cpu::index(A.dtype, B.dtype)](&A, &B, &Z, false, false);

    float Z_expected_data[2 * 2 * 2] = {
        1*9+2*7, 1*8+2*6,
        3*9+4*7, 3*8+4*6,
        5*5+6*3, 5*4+6*2,
        7*5+8*3, 7*4+8*2
    };

    float epsilon = 1e-5f;
    for (size_t b = 0; b < shape_Z[0]; ++b) {
        for (size_t i = 0; i < shape_Z[1]; ++i) {
            for (size_t j = 0; j < shape_Z[2]; ++j) {
                size_t idx = b * strides_Z[0] + i * strides_Z[1] + j * strides_Z[2];
                ASSERT_NEAR(((float*)Z.storage.address)[idx], Z_expected_data[b * 4 + i * 2 + j], epsilon)
                    << "Mismatch at batch " << b << ", (" << i << "," << j << ")";
            }
        }
    }
}

TEST(MatmulTests, Rank4_SecondTransposed) {
    const size_t batch1 = 2, batch2 = 2;
    const size_t M = 2, K = 4, N = 3;
 
    float X_data[batch1 * batch2 * M * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8
    };
 
    float Y_data[batch1 * batch2 * N * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16,
        17,18,19,20,
        21,22,23,24,
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4
    };

    float Z_data[batch1 * batch2 * M * N] = {0};
 
    size_t shape_X[4] = {batch1, batch2, M, K};
    size_t shape_Y[4] = {batch1, batch2, N, K};
    size_t shape_Z[4] = {batch1, batch2, M, N};

    size_t strides_X[4] = {batch2 * M * K, M * K, K, 1};
    size_t strides_Y[4] = {batch2 * N * K, N * K, K, 1};
    size_t strides_Z[4] = {batch2 * M * N, M * N, N, 1};

    tensor_t X = {
        .rank = 4, 
        .shape = shape_X, 
        .strides = strides_X, 
        .storage = {.address = X_data, .nbytes = sizeof(X_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Y = {
        .rank = 4, 
        .shape = shape_Y, 
        .strides = strides_Y, 
        .storage = {.address = Y_data, .nbytes = sizeof(Y_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };
    tensor_t Z = {
        .rank = 4, 
        .shape = shape_Z, 
        .strides = strides_Z, 
        .storage = {.address = Z_data, .nbytes = sizeof(Z_data), .resource = {0}}, 
        .offset = 0, 
        .dtype = float32
    };

    memset(Z_data, 0, sizeof(Z_data));
 
    cpu::matmul::kernels[cpu::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, true);
 
    float Z_expected[batch1 * batch2 * M * N] = {
        30.f, 70.f, 110.f,
        70.f, 174.f, 278.f,
        150.f, 190.f, 230.f,
        382.f, 486.f, 590.f,
        4.f, 6.f, 10.f,
        12.f, 14.f, 26.f,
        20.f, 30.f, 40.f,
        52.f, 78.f, 104.f
    };

    float epsilon = 1e-5f;
    for (size_t i = 0; i < batch1 * batch2 * M * N; ++i) {
        ASSERT_NEAR(Z_data[i], Z_expected[i], epsilon) << "Mismatch at flat index " << i;
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