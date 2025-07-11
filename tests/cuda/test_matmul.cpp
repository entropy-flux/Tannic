#ifdef CUDA

#include <gtest/gtest.h> 
#include <cstring>
#include <cmath>

#include "core/types.h"
#include "core/tensor.h" 
#include "cuda/matmul-op.cuh"     

class CudaMatmulTests : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream);
    }

    void TearDown() override {
        cudaStreamDestroy(stream);
    }

    cudaStream_t stream;
};

TEST_F(CudaMatmulTests, Basic) {
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

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, sizeof(X_data));
    cudaMalloc(&d_Y, sizeof(Y_data));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_X, X_data, sizeof(X_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_data, sizeof(Y_data), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_X[2] = {3, 4};
    size_t shape_Y[2] = {4, 5};
    size_t shape_Z[2] = {3, 5};

    size_t strides_X[2] = {4, 1};
    size_t strides_Y[2] = {5, 1};
    size_t strides_Z[2] = {5, 1};

    tensor_t X = {.rank = 2, .shape = shape_X, .strides = strides_X, .offset = 0, .data = d_X, .dtype = float32};
    tensor_t Y = {.rank = 2, .shape = shape_Y, .strides = strides_Y, .offset = 0, .data = d_Y, .dtype = float32};
    tensor_t Z = {.rank = 2, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA matmul
    cuda::matmul::kernels[cuda::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, false, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(Z_data[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
}

TEST_F(CudaMatmulTests, FirstTransposed) {
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    float Y_data_fixed[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };

    float Z_data[3 * 3] = {0};
    float Z_expected_data[3 * 3] = {
        47.f, 52.f, 57.f,
        64.f, 71.f, 78.f,
        81.f, 90.f, 99.f
    };

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, sizeof(X_data));
    cudaMalloc(&d_Y, sizeof(Y_data_fixed));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_X, X_data, sizeof(X_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_data_fixed, sizeof(Y_data_fixed), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_X[2] = {2, 3};  
    size_t shape_Y[2] = {2, 3}; 
    size_t shape_Z[2] = {3, 3};  

    size_t strides_X[2] = {3, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {3, 1};

    tensor_t X = {.rank = 2, .shape = shape_X, .strides = strides_X, .offset = 0, .data = d_X, .dtype = float32};
    tensor_t Y = {.rank = 2, .shape = shape_Y, .strides = strides_Y, .offset = 0, .data = d_Y, .dtype = float32};
    tensor_t Z = {.rank = 2, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA matmul with first transposed
    cuda::matmul::kernels[cuda::index(X.dtype,Y.dtype)](&X, &Y, &Z, true, false, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(Z_data[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
}

TEST_F(CudaMatmulTests, SecondTransposed) {
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    float Y_data[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };

    float Z_data[2 * 2] = {0};
    float Z_expected_data[2 * 2] = {
        50.f, 68.f,
        122.f, 167.f
    };

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, sizeof(X_data));
    cudaMalloc(&d_Y, sizeof(Y_data));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_X, X_data, sizeof(X_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_data, sizeof(Y_data), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_X[2] = {2, 3};
    size_t shape_Y[2] = {2, 3};
    size_t shape_Z[2] = {2, 2};

    size_t strides_X[2] = {3, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {2, 1};

    tensor_t X = {.rank = 2, .shape = shape_X, .strides = strides_X, .offset = 0, .data = d_X, .dtype = float32};
    tensor_t Y = {.rank = 2, .shape = shape_Y, .strides = strides_Y, .offset = 0, .data = d_Y, .dtype = float32};
    tensor_t Z = {.rank = 2, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA matmul with second transposed
    cuda::matmul::kernels[cuda::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, true, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(Z_data[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
} 

TEST_F(CudaMatmulTests, BothTransposed) {
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

    float Z_expected_data[2 * 2] = {
        50.f, 68.f,
        122.f, 167.f
    };

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, sizeof(X_data));
    cudaMalloc(&d_Y, sizeof(Y_data));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_X, X_data, sizeof(X_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_data, sizeof(Y_data), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_X[2] = {3, 2};
    size_t shape_Y[2] = {2, 3};
    size_t shape_Z[2] = {2, 2};

    size_t strides_X[2] = {2, 1};
    size_t strides_Y[2] = {3, 1};
    size_t strides_Z[2] = {2, 1};

    tensor_t X = {.rank = 2, .shape = shape_X, .strides = strides_X, .offset = 0, .data = d_X, .dtype = float32};
    tensor_t Y = {.rank = 2, .shape = shape_Y, .strides = strides_Y, .offset = 0, .data = d_Y, .dtype = float32};
    tensor_t Z = {.rank = 2, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA matmul with both transposed
    cuda::matmul::kernels[cuda::index(X.dtype,Y.dtype)](&X, &Y, &Z, true, true, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t i = 0; i < shape_Z[0]; ++i) {
        for (size_t j = 0; j < shape_Z[1]; ++j) {
            size_t idx = i * strides_Z[0] + j * strides_Z[1];
            ASSERT_NEAR(Z_data[idx], Z_expected_data[i * shape_Z[1] + j], epsilon)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
}
 

TEST_F(CudaMatmulTests, Batched) { 
    float A_data[2 * 2 * 2] = {
        // batch 0
        1.f, 2.f,
        3.f, 4.f,
        // batch 1
        5.f, 6.f,
        7.f, 8.f
    };

    float B_data[2 * 2 * 2] = {
        // batch 0
        9.f, 8.f,
        7.f, 6.f,
        // batch 1
        5.f, 4.f,
        3.f, 2.f
    };

    float Z_data[2 * 2 * 2] = {0};
    float Z_expected_data[2 * 2 * 2] = {
        // batch 0: A[0] @ B[0]
        1*9+2*7, 1*8+2*6,
        3*9+4*7, 3*8+4*6,
        // batch 1: A[1] @ B[1]
        5*5+6*3, 5*4+6*2,
        7*5+8*3, 7*4+8*2
    };

    // Allocate device memory
    float *d_A, *d_B, *d_Z;
    cudaMalloc(&d_A, sizeof(A_data));
    cudaMalloc(&d_B, sizeof(B_data));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_A, A_data, sizeof(A_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_data, sizeof(B_data), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_A[3] = {2, 2, 2};
    size_t shape_B[3] = {2, 2, 2};
    size_t shape_Z[3] = {2, 2, 2};

    size_t strides_A[3] = {4, 2, 1};
    size_t strides_B[3] = {4, 2, 1};
    size_t strides_Z[3] = {4, 2, 1};

    tensor_t A = {.rank = 3, .shape = shape_A, .strides = strides_A, .offset = 0, .data = d_A, .dtype = float32};
    tensor_t B = {.rank = 3, .shape = shape_B, .strides = strides_B, .offset = 0, .data = d_B, .dtype = float32};
    tensor_t Z = {.rank = 3, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA batched matmul
    cuda::matmul::kernels[cuda::index(A.dtype, B.dtype)](&A, &B, &Z, false, false, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t b = 0; b < shape_Z[0]; ++b) {
        for (size_t i = 0; i < shape_Z[1]; ++i) {
            for (size_t j = 0; j < shape_Z[2]; ++j) {
                size_t idx = b * strides_Z[0] + i * strides_Z[1] + j * strides_Z[2];
                ASSERT_NEAR(Z_data[idx], Z_expected_data[b * 4 + i * 2 + j], epsilon)
                    << "Mismatch at batch " << b << ", (" << i << "," << j << ")";
            }
        }
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Z);
}

TEST_F(CudaMatmulTests, Rank4_SecondTransposed) {
    const size_t batch1 = 2, batch2 = 2;
    const size_t M = 2, K = 4, N = 3;
 
    float X_data[batch1 * batch2 * M * K] = {
        // batch1=0, batch2=0
        1, 2, 3, 4,
        5, 6, 7, 8,
        // batch1=0, batch2=1
        1, 2, 3, 4,
        5, 6, 7, 8,
        // batch1=1, batch2=0
        1, 2, 3, 4,
        5, 6, 7, 8,
        // batch1=1, batch2=1
        1, 2, 3, 4,
        5, 6, 7, 8
    };
 
    float Y_data[batch1 * batch2 * N * K] = {
        // batch1=0, batch2=0
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        // batch1=0, batch2=1
        13,14,15,16,
        17,18,19,20,
        21,22,23,24,
        // batch1=1, batch2=0
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 1, 1, 1,
        // batch1=1, batch2=1
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4
    };

    float Z_data[batch1 * batch2 * M * N] = {0};
    float Z_expected[batch1 * batch2 * M * N] = {
        // batch1=0, batch2=0
        30.f, 70.f, 110.f,
        70.f, 174.f, 278.f,
        // batch1=0, batch2=1
        150.f, 190.f, 230.f,
        382.f, 486.f, 590.f,
        // batch1=1, batch2=0
        4.f, 6.f, 10.f,
        12.f, 14.f, 26.f,
        // batch1=1, batch2=1
        20.f, 30.f, 40.f,
        52.f, 78.f, 104.f
    };

    // Allocate device memory
    float *d_X, *d_Y, *d_Z;
    cudaMalloc(&d_X, sizeof(X_data));
    cudaMalloc(&d_Y, sizeof(Y_data));
    cudaMalloc(&d_Z, sizeof(Z_data));

    // Copy data to device
    cudaMemcpy(d_X, X_data, sizeof(X_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y_data, sizeof(Y_data), cudaMemcpyHostToDevice);
    cudaMemset(d_Z, 0, sizeof(Z_data));

    size_t shape_X[4] = {batch1, batch2, M, K};
    size_t shape_Y[4] = {batch1, batch2, N, K};
    size_t shape_Z[4] = {batch1, batch2, M, N};

    size_t strides_X[4] = {batch2 * M * K, M * K, K, 1};
    size_t strides_Y[4] = {batch2 * N * K, N * K, K, 1};
    size_t strides_Z[4] = {batch2 * M * N, M * N, N, 1};

    tensor_t X = {.rank = 4, .shape = shape_X, .strides = strides_X, .offset = 0, .data = d_X, .dtype = float32};
    tensor_t Y = {.rank = 4, .shape = shape_Y, .strides = strides_Y, .offset = 0, .data = d_Y, .dtype = float32};
    tensor_t Z = {.rank = 4, .shape = shape_Z, .strides = strides_Z, .offset = 0, .data = d_Z, .dtype = float32};

    // Run CUDA matmul with second transposed
    cuda::matmul::kernels[cuda::index(X.dtype,Y.dtype)](&X, &Y, &Z, false, true, stream);
    cudaStreamSynchronize(stream);

    // Copy result back
    cudaMemcpy(Z_data, d_Z, sizeof(Z_data), cudaMemcpyDeviceToHost);

    // Verify results
    float epsilon = 1e-5f;
    for (size_t i = 0; i < batch1 * batch2 * M * N; ++i) {
        ASSERT_NEAR(Z_data[i], Z_expected[i], epsilon) << "Mismatch at flat index " << i;
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
}

#endif