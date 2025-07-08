#include <gtest/gtest.h> 
#include <cmath>
#include <vector>

#include "core/types.h"
#include "core/tensor.h"
#include "cuda/unary-ops.cuh"  // Assuming this is the header with your kernels

class TestUnaryOpsCuda : public ::testing::Test {
public:
    float h_A_data[6];  // Host
    float* d_A_data{};  // Device
    tensor_t A;

    size_t shape[3] = {2, 1, 3};
    size_t strides[3] = {3, 3, 1};
    const type dtype = float32;

protected:
    void SetUp() override {
        for (int i = 0; i < 6; ++i)
            h_A_data[i] = static_cast<float>(i + 1);  // 1 to 6

        cudaMalloc(&d_A_data, sizeof(float) * 6);
        cudaMemcpy(d_A_data, h_A_data, sizeof(float) * 6, cudaMemcpyHostToDevice);

        A.rank = 3;
        A.shape = shape;
        A.strides = strides;
        A.offset = 0;
        A.address = d_A_data;
        A.dtype = dtype;
    }

    void TearDown() override {
        cudaFree(d_A_data);
    }

    void checkUnaryOp(auto kernel_table, auto op_func, const float expected[2][1][3]) {
        tensor_t B = A;  // In-place op
        kernel_table[A.dtype](&A, &B, op_func, nullptr);

        cudaMemcpy(h_A_data, d_A_data, sizeof(float) * 6, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 1; ++j)
        for (int k = 0; k < 3; ++k) {
            int idx = i * 3 + j * 3 + k;
            EXPECT_FLOAT_EQ(h_A_data[idx], expected[i][j][k])
                << "Mismatch at [" << i << "][" << j << "][" << k << "]";
        }
    }
};

TEST_F(TestUnaryOpsCuda, Negation) {
    float expected[2][1][3] = {{{-1.0f, -2.0f, -3.0f}}, {{-4.0f, -5.0f, -6.0f}}};
    checkUnaryOp(cuda::negation_op::kernels, cuda::negation_op::Negation{}, expected);
}

TEST_F(TestUnaryOpsCuda, Log) {
    float expected[2][1][3] = {{{std::log(1.0f), std::log(2.0f), std::log(3.0f)}},
                               {{std::log(4.0f), std::log(5.0f), std::log(6.0f)}}};
    checkUnaryOp(cuda::log_op::kernels, cuda::log_op::Log{}, expected);
}

TEST_F(TestUnaryOpsCuda, Exp) {
    float expected[2][1][3] = {{{std::exp(1.0f), std::exp(2.0f), std::exp(3.0f)}},
                               {{std::exp(4.0f), std::exp(5.0f), std::exp(6.0f)}}};
    checkUnaryOp(cuda::exp_op::kernels, cuda::exp_op::Exp{}, expected);
}

TEST_F(TestUnaryOpsCuda, Sqrt) {
    float expected[2][1][3] = {{{std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f)}},
                               {{std::sqrt(4.0f), std::sqrt(5.0f), std::sqrt(6.0f)}}};
    checkUnaryOp(cuda::sqrt_op::kernels, cuda::sqrt_op::Sqrt{}, expected);
}

TEST_F(TestUnaryOpsCuda, Abs) {
    for (int i = 0; i < 6; ++i)
        h_A_data[i] = -1.0f * (i + 1);
    cudaMemcpy(d_A_data, h_A_data, sizeof(float) * 6, cudaMemcpyHostToDevice);

    float expected[2][1][3] = {{{1.0f, 2.0f, 3.0f}}, {{4.0f, 5.0f, 6.0f}}};
    checkUnaryOp(cuda::abs_op::kernels, cuda::abs_op::Abs{}, expected);
}

TEST_F(TestUnaryOpsCuda, Sin) {
    float expected[2][1][3] = {{{std::sin(1.0f), std::sin(2.0f), std::sin(3.0f)}},
                               {{std::sin(4.0f), std::sin(5.0f), std::sin(6.0f)}}};
    checkUnaryOp(cuda::sin_op::kernels, cuda::sin_op::Sin{}, expected);
}

TEST_F(TestUnaryOpsCuda, Cos) {
    float expected[2][1][3] = {{{std::cos(1.0f), std::cos(2.0f), std::cos(3.0f)}},
                               {{std::cos(4.0f), std::cos(5.0f), std::cos(6.0f)}}};
    checkUnaryOp(cuda::cos_op::kernels, cuda::cos_op::Cos{}, expected);
}

TEST_F(TestUnaryOpsCuda, Tan) {
    float expected[2][1][3] = {{{std::tan(1.0f), std::tan(2.0f), std::tan(3.0f)}},
                               {{std::tan(4.0f), std::tan(5.0f), std::tan(6.0f)}}};
    checkUnaryOp(cuda::tan_op::kernels, cuda::tan_op::Tan{}, expected);
}

TEST_F(TestUnaryOpsCuda, Sinh) {
    float expected[2][1][3] = {{{std::sinh(1.0f), std::sinh(2.0f), std::sinh(3.0f)}},
                               {{std::sinh(4.0f), std::sinh(5.0f), std::sinh(6.0f)}}};
    checkUnaryOp(cuda::sinh_op::kernels, cuda::sinh_op::Sinh{}, expected);
}

TEST_F(TestUnaryOpsCuda, Cosh) {
    float expected[2][1][3] = {{{std::cosh(1.0f), std::cosh(2.0f), std::cosh(3.0f)}},
                               {{std::cosh(4.0f), std::cosh(5.0f), std::cosh(6.0f)}}};
    checkUnaryOp(cuda::cosh_op::kernels, cuda::cosh_op::Cosh{}, expected);
}

TEST_F(TestUnaryOpsCuda, Tanh) {
    float expected[2][1][3] = {{{std::tanh(1.0f), std::tanh(2.0f), std::tanh(3.0f)}},
                               {{std::tanh(4.0f), std::tanh(5.0f), std::tanh(6.0f)}}};
    checkUnaryOp(cuda::tanh_op::kernels, cuda::tanh_op::Tanh{}, expected);
}
