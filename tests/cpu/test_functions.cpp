 #include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "core/types.h"
#include "core/tensor.h"
#include "cpu/unary-ops.hpp"

class TestUnaryOps : public ::testing::Test {
public:
    float A_data[2 * 1 * 3];
    size_t shape_A[3] = {2, 1, 3};
    size_t strides_A[3] = {3, 3, 1};

    tensor_t A;
    const type dtype_float = float32;

protected:
    void SetUp() override {
        for (int i = 0; i < 6; ++i)
            A_data[i] = static_cast<float>(i + 1);  // 1 to 6

        A.rank = 3;
        A.shape = shape_A;
        A.strides = strides_A;
        A.offset = 0;
        A.address = static_cast<void*>(A_data);
        A.dtype = dtype_float;
    }
};


TEST_F(TestUnaryOps, Negation) {
    cpu::negation::kernels[A.dtype](&A, &A, cpu::negation::Negation{});

    float expected[2][1][3] = {
        {
            { -1.0f, -2.0f, -3.0f }
        },
        {
            { -4.0f, -5.0f, -6.0f }
        }
    };

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 1; ++j)
            for (int k = 0; k < 3; ++k) {
                int idx = i * 3 + j * 3 + k;
                EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k])
                    << "Mismatch at A[" << i << "][" << j << "][" << k << "]";
            }
}


TEST_F(TestUnaryOps, Log) {
    cpu::log::kernels[A.dtype](&A, &A, cpu::log::Log{});
    float expected[2][1][3] = {
        { { std::log(1.0f), std::log(2.0f), std::log(3.0f) } },
        { { std::log(4.0f), std::log(5.0f), std::log(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Exp) {
    cpu::exp::kernels[A.dtype](&A, &A, cpu::exp::Exp{});
    float expected[2][1][3] = {
        { { std::exp(1.0f), std::exp(2.0f), std::exp(3.0f) } },
        { { std::exp(4.0f), std::exp(5.0f), std::exp(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Sqrt) {
    cpu::sqrt::kernels[A.dtype](&A, &A, cpu::sqrt::Sqrt{});
    float expected[2][1][3] = {
        { { std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f) } },
        { { std::sqrt(4.0f), std::sqrt(5.0f), std::sqrt(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Abs) {
    for (int i = 0; i < 6; ++i)
        A_data[i] = static_cast<float>(-1 * (i + 1));  // -1 to -6

    cpu::abs::kernels[A.dtype](&A, &A, cpu::abs::Abs{});

    float expected[2][1][3] = {
        { { 1.0f, 2.0f, 3.0f } },
        { { 4.0f, 5.0f, 6.0f } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Sin) {
    cpu::sin::kernels[A.dtype](&A, &A, cpu::sin::Sin{});
    float expected[2][1][3] = {
        { { std::sin(1.0f), std::sin(2.0f), std::sin(3.0f) } },
        { { std::sin(4.0f), std::sin(5.0f), std::sin(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Cos) {
    cpu::cos::kernels[A.dtype](&A, &A, cpu::cos::Cos{});
    float expected[2][1][3] = {
        { { std::cos(1.0f), std::cos(2.0f), std::cos(3.0f) } },
        { { std::cos(4.0f), std::cos(5.0f), std::cos(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Tan) {
    cpu::tan::kernels[A.dtype](&A, &A, cpu::tan::Tan{});
    float expected[2][1][3] = {
        { { std::tan(1.0f), std::tan(2.0f), std::tan(3.0f) } },
        { { std::tan(4.0f), std::tan(5.0f), std::tan(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Sinh) {
    cpu::sinh::kernels[A.dtype](&A, &A, cpu::sinh::Sinh{});
    float expected[2][1][3] = {
        { { std::sinh(1.0f), std::sinh(2.0f), std::sinh(3.0f) } },
        { { std::sinh(4.0f), std::sinh(5.0f), std::sinh(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Cosh) {
    cpu::cosh::kernels[A.dtype](&A, &A, cpu::cosh::Cosh{});
    float expected[2][1][3] = {
        { { std::cosh(1.0f), std::cosh(2.0f), std::cosh(3.0f) } },
        { { std::cosh(4.0f), std::cosh(5.0f), std::cosh(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}

TEST_F(TestUnaryOps, Tanh) {
    cpu::tanh::kernels[A.dtype](&A, &A, cpu::tanh::Tanh{});
    float expected[2][1][3] = {
        { { std::tanh(1.0f), std::tanh(2.0f), std::tanh(3.0f) } },
        { { std::tanh(4.0f), std::tanh(5.0f), std::tanh(6.0f) } }
    };
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 1; ++j)
    for (int k = 0; k < 3; ++k) {
        int idx = i * 3 + j * 3 + k;
        EXPECT_FLOAT_EQ(A_data[idx], expected[i][j][k]);
    }
}
