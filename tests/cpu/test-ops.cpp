#include <gtest/gtest.h>
#include <vector>
#include <numeric>

#include "core/types.h"
#include "core/tensor.h"
#include "cpu/unary-ops.hpp"
#include "cpu/binary-ops.hpp"

class TestBinaryOps : public ::testing::Test {
public:
    size_t shape_A_arr[3];
    size_t shape_B_arr[3];
    size_t shape_C_arr[3];
    size_t shape_D_arr[3];

    size_t strides_A_arr[3];
    size_t strides_B_arr[3];
    size_t strides_C_arr[3];
    size_t strides_D_arr[3];

    float A_data[2*1*3];
    float B_data[1*4*3];
    float C_data[2*4*3];
    float D_data[2*1*3];

    storage_t A_storage, B_storage, C_storage, D_storage;
    tensor_t A, B, C, D;

    size_t orig_strides_A[3];
    size_t orig_strides_B[3];
    size_t orig_strides_C[3];
    size_t orig_strides_D[3];

    const type dtype_float = float32;  
    
    void broadcasted_strides(tensor_t* tensor, const tensor_t* output) const {
        for (uint8_t i = 0; i < output->rank; ++i) {
            tensor->strides[i] = (tensor->shape[i] == 1) ? 0 : tensor->strides[i];
        }
    }

protected:
    void SetUp() override {
        uint8_t rank = 3;

        // Initialize shapes
        shape_A_arr[0] = 2; shape_A_arr[1] = 1; shape_A_arr[2] = 3;
        shape_B_arr[0] = 1; shape_B_arr[1] = 4; shape_B_arr[2] = 3;
        shape_C_arr[0] = 2; shape_C_arr[1] = 4; shape_C_arr[2] = 3;
        shape_D_arr[0] = 2; shape_D_arr[1] = 1; shape_D_arr[2] = 3;

        // Initialize strides
        strides_A_arr[0] = 3; strides_A_arr[1] = 3; strides_A_arr[2] = 1;
        strides_B_arr[0] = 12; strides_B_arr[1] = 3; strides_B_arr[2] = 1;
        strides_C_arr[0] = 12; strides_C_arr[1] = 3; strides_C_arr[2] = 1;
        strides_D_arr[0] = 12; strides_D_arr[1] = 3; strides_D_arr[2] = 1;

        // Save original strides
        memcpy(orig_strides_A, strides_A_arr, sizeof(strides_A_arr));
        memcpy(orig_strides_B, strides_B_arr, sizeof(strides_B_arr));
        memcpy(orig_strides_C, strides_C_arr, sizeof(strides_C_arr));
        memcpy(orig_strides_D, strides_D_arr, sizeof(strides_D_arr));

        // Initialize data
        for (int i = 0; i < 6; ++i) A_data[i] = static_cast<float>(i);
        for (int i = 0; i < 12; ++i) B_data[i] = static_cast<float>(i * 10);
        for (int i = 0; i < 24; ++i) C_data[i] = 0.0f;
        for (int i = 0; i < 6; ++i) D_data[i] = 0.0f;

        // Initialize storage
        A_storage = {.address = A_data, .nbytes = sizeof(A_data), .resource = {0}};
        B_storage = {.address = B_data, .nbytes = sizeof(B_data), .resource = {0}};
        C_storage = {.address = C_data, .nbytes = sizeof(C_data), .resource = {0}};
        D_storage = {.address = D_data, .nbytes = sizeof(D_data), .resource = {0}};

        // Initialize tensors
        A = {
            .rank = rank,
            .shape = shape_A_arr,
            .strides = strides_A_arr,
            .storage = A_storage,
            .offset = 0,
            .dtype = dtype_float
        };

        B = {
            .rank = rank,
            .shape = shape_B_arr,
            .strides = strides_B_arr,
            .storage = B_storage,
            .offset = 0,
            .dtype = dtype_float
        };

        C = {
            .rank = rank,
            .shape = shape_C_arr,
            .strides = strides_C_arr,
            .storage = C_storage,
            .offset = 0,
            .dtype = dtype_float
        };

        D = {
            .rank = rank,
            .shape = shape_D_arr,
            .strides = strides_D_arr,
            .storage = D_storage,
            .offset = 0,
            .dtype = dtype_float
        };
    }

    void reset_strides() {
        memcpy(strides_A_arr, orig_strides_A, sizeof(strides_A_arr));
        memcpy(strides_B_arr, orig_strides_B, sizeof(strides_B_arr));
        memcpy(strides_C_arr, orig_strides_C, sizeof(strides_C_arr));
        memcpy(strides_D_arr, orig_strides_D, sizeof(strides_D_arr));
    }
};
 

TEST_F(TestBinaryOps, Addition) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);
 
    cpu::addition::kernels[cpu::index(A.dtype,B.dtype)](&A, &B, &C, cpu::addition::Addition{});

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

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestBinaryOps, Multiplication) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);

    cpu::multiplication::kernels[cpu::index(A.dtype, B.dtype)](&A, &B, &C, cpu::multiplication::Multiplication{});

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

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
            }
}

TEST_F(TestBinaryOps, Subtraction) {
    broadcasted_strides(&A, &C);
    broadcasted_strides(&B, &C);

    cpu::subtraction::kernels[cpu::index(A.dtype, B.dtype)](&A, &B, &C, cpu::subtraction::Subtraction{});

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

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 12 + j * 3 + k;
                EXPECT_FLOAT_EQ(C_data[idx], expected[i][j][k]) << "Mismatch at C[" << i << "][" << j << "][" << k << "]";
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
        self.shape_C = (2, 4, 3)
        self.shape_D = (2, 1, 3)

        # Data initialization matching your C++ arrays
        self.A = np.arange(6, dtype=np.float32).reshape(self.shape_A)
        self.B = (np.arange(12, dtype=np.float32) * 10).reshape(self.shape_B)
        self.C = np.zeros(self.shape_C, dtype=np.float32)
        self.D = np.zeros(self.shape_D, dtype=np.float32)

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