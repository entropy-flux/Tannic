#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "tensor.hpp"
#include "functions.hpp"

using namespace tannic;

class TestFunctions : public ::testing::Test {
protected:
    Tensor A;

    TestFunctions() : A(float32, Shape(2, 1, 3)) { 
        A.initialize();
    }
     
    void SetUp() override { 
        float* data = reinterpret_cast<float*>(A.bytes());
        for (int i = 0; i < 6; ++i) {
            data[i] = static_cast<float>(i + 1);
        }
    }
};

TEST_F(TestFunctions, Log) {
    Tensor result = log(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::log(1.0f), std::log(2.0f), std::log(3.0f),
        std::log(4.0f), std::log(5.0f), std::log(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Exp) {
    Tensor result = exp(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::exp(1.0f), std::exp(2.0f), std::exp(3.0f),
        std::exp(4.0f), std::exp(5.0f), std::exp(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Sqrt) {
    Tensor result = sqrt(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f),
        std::sqrt(4.0f), std::sqrt(5.0f), std::sqrt(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Rsqrt) {
    Tensor result = rsqrt(A, 1e-6);
    const float* data = reinterpret_cast<const float*>(result.bytes());

    const float expected[6] = {
        1.0f / std::sqrt(1.0f + 1e-6f),
        1.0f / std::sqrt(2.0f + 1e-6f),
        1.0f / std::sqrt(3.0f + 1e-6f),
        1.0f / std::sqrt(4.0f + 1e-6f),
        1.0f / std::sqrt(5.0f + 1e-6f),
        1.0f / std::sqrt(6.0f + 1e-6f)
    };

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
} 

TEST_F(TestFunctions, Abs) { 
    float* original_data = reinterpret_cast<float*>(A.bytes());
    for (int i = 0; i < 6; ++i) {
        original_data[i] = -original_data[i];
    }

    Tensor result = abs(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Sin) {
    Tensor result = sin(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::sin(1.0f), std::sin(2.0f), std::sin(3.0f),
        std::sin(4.0f), std::sin(5.0f), std::sin(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Cos) {
    Tensor result = cos(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::cos(1.0f), std::cos(2.0f), std::cos(3.0f),
        std::cos(4.0f), std::cos(5.0f), std::cos(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Tan) {
    Tensor result = tan(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::tan(1.0f), std::tan(2.0f), std::tan(3.0f),
        std::tan(4.0f), std::tan(5.0f), std::tan(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Sinh) {
    Tensor result = sinh(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::sinh(1.0f), std::sinh(2.0f), std::sinh(3.0f),
        std::sinh(4.0f), std::sinh(5.0f), std::sinh(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Cosh) {
    Tensor result = cosh(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::cosh(1.0f), std::cosh(2.0f), std::cosh(3.0f),
        std::cosh(4.0f), std::cosh(5.0f), std::cosh(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST_F(TestFunctions, Tanh) {
    Tensor result = tanh(A);
    const float* data = reinterpret_cast<const float*>(result.bytes());
    
    const float expected[6] = {
        std::tanh(1.0f), std::tanh(2.0f), std::tanh(3.0f),
        std::tanh(4.0f), std::tanh(5.0f), std::tanh(6.0f)
    };
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}
