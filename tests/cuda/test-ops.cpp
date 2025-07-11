#ifdef CUDA

#include <gtest/gtest.h> 

#include "core/types.h"
#include "core/tensor.h"
#include "cuda/binary-ops.cuh" // CUDA kernels assumed here

class TestCUDABinaryOps : public ::testing::Test {
public:
    float *A_data_d, *B_data_d, *C_data_d;
    float A_data[2 * 1 * 3];
    float B_data[1 * 4 * 3];
    float C_data[2 * 4 * 3];

    size_t shape_A[3] = {2, 1, 3};
    size_t shape_B[3] = {1, 4, 3};
    size_t shape_C[3] = {2, 4, 3};

    size_t strides_A[3] = {3, 3, 1};
    size_t strides_B[3] = {12, 3, 1};
    size_t strides_C[3] = {12, 3, 1};

    storage_t A_storage, B_storage, C_storage;
    tensor_t A, B, C;

    const type dtype_float = float32;

    void broadcasted_strides(tensor_t* tensor, const tensor_t* output) const {
        for (uint8_t i = 0; i < output->rank; ++i)
            tensor->strides[i] = (tensor->shape[i] == 1) ? 0 : tensor->strides[i];
    }

protected:
    void SetUp() override {
        // Initialize data
        for (int i = 0; i < 6; ++i) A_data[i] = static_cast<float>(i);
        for (int i = 0; i < 12; ++i) B_data[i] = static_cast<float>(i * 10);
        for (int i = 0; i < 24; ++i) C_data[i] = 0.0f;

        // Allocate device memory
        cudaMalloc(&A_data_d, sizeof(float) * 6);
        cudaMalloc(&B_data_d, sizeof(float) * 12);
        cudaMalloc(&C_data_d, sizeof(float) * 24);

        // Copy to device
        cudaMemcpy(A_data_d, A_data, sizeof(float) * 6, cudaMemcpyHostToDevice);
        cudaMemcpy(B_data_d, B_data, sizeof(float) * 12, cudaMemcpyHostToDevice);

        // Initialize storage
        A_storage = {.address = A_data_d, .nbytes = sizeof(float) * 6, .resource = {0}};
        B_storage = {.address = B_data_d, .nbytes = sizeof(float) * 12, .resource = {0}};
        C_storage = {.address = C_data_d, .nbytes = sizeof(float) * 24, .resource = {0}};

        // Initialize tensors
        A = {.rank = 3, .shape = shape_A, .strides = strides_A, .storage = &A_storage, .offset = 0, .dtype = dtype_float};
        B = {.rank = 3, .shape = shape_B, .strides = strides_B, .storage = &B_storage, .offset = 0, .dtype = dtype_float};
        C = {.rank = 3, .shape = shape_C, .strides = strides_C, .storage = &C_storage, .offset = 0, .dtype = dtype_float};
    }

    void TearDown() override {
        cudaFree(A_data_d);
        cudaFree(B_data_d);
        cudaFree(C_data_d);
    }

    void fetch_output() {
        cudaMemcpy(C_data, C_data_d, sizeof(float) * 24, cudaMemcpyDeviceToHost);
    }
};

TEST_F(TestCUDABinaryOps, Addition) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);

    // Pass null stream (0) explicitly
    cuda::addition::kernels[cuda::index(A.dtype, B.dtype)](&A, &B, &C, cuda::addition::Addition{}, 0);

    fetch_output();

    float expected[2][4][3] = {
        {{0.0f, 11.0f, 22.0f}, {30.0f, 41.0f, 52.0f}, {60.0f, 71.0f, 82.0f}, {90.0f, 101.0f, 112.0f}},
        {{3.0f, 14.0f, 25.0f}, {33.0f, 44.0f, 55.0f}, {63.0f, 74.0f, 85.0f}, {93.0f, 104.0f, 115.0f}}
    };

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestCUDABinaryOps, Multiplication) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);

    // Pass null stream (0) explicitly
    cuda::multiplication::kernels[cuda::index(A.dtype, B.dtype)](&A, &B, &C, cuda::multiplication::Multiplication{}, 0);

    fetch_output();

    float expected[2][4][3] = {
        {{0.0f, 10.0f, 40.0f}, {0.0f, 40.0f, 100.0f}, {0.0f, 70.0f, 160.0f}, {0.0f, 100.0f, 220.0f}},
        {{0.0f, 40.0f, 100.0f}, {90.0f, 160.0f, 250.0f}, {180.0f, 280.0f, 400.0f}, {270.0f, 400.0f, 550.0f}}
    };

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestCUDABinaryOps, Subtraction) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);

    // Pass null stream (0) explicitly
    cuda::subtraction::kernels[cuda::index(A.dtype, B.dtype)](&A, &B, &C, cuda::subtraction::Subtraction{}, 0);

    fetch_output();

    float expected[2][4][3] = {
        {{0.0f, -9.0f, -18.0f}, {-30.0f, -39.0f, -48.0f}, {-60.0f, -69.0f, -78.0f}, {-90.0f, -99.0f, -108.0f}},
        {{3.0f, -6.0f, -15.0f}, {-27.0f, -36.0f, -45.0f}, {-57.0f, -66.0f, -75.0f}, {-87.0f, -96.0f, -105.0f}}
    };

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "C[" << i << "][" << j << "][" << k << "]";
            }
}

#endif