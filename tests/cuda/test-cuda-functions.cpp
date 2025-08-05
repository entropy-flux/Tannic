#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>  // For CUDA memory operations
#include "Tensor.hpp"
#include "Functions.hpp"

using namespace tannic;

class TestCUDAFunctions : public ::testing::Test {
protected:
    Tensor A;

    TestCUDAFunctions() : A(float32, Shape(2, 1, 3)) {
        A.initialize(Device(0));  // Initialize on GPU
    }

    void SetUp() override { 
        float host_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        cudaMemcpy(A.bytes(), host_data, 6 * sizeof(float), cudaMemcpyHostToDevice);
    }
 
    void compareWithExpected(const Tensor& result, const float expected[6]) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float cpu_data[6];
         
        cudaMemcpy(cpu_data, gpu_data, 6 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(cpu_data[i], expected[i]);
        }
    }
};

TEST_F(TestCUDAFunctions, Log) {
    Tensor result = log(A);
    const float expected[6] = {
        std::log(1.0f), std::log(2.0f), std::log(3.0f),
        std::log(4.0f), std::log(5.0f), std::log(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Exp) {
    Tensor result = exp(A);
    const float expected[6] = {
        std::exp(1.0f), std::exp(2.0f), std::exp(3.0f),
        std::exp(4.0f), std::exp(5.0f), std::exp(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Sqrt) {
    Tensor result = sqrt(A);
    const float expected[6] = {
        std::sqrt(1.0f), std::sqrt(2.0f), std::sqrt(3.0f),
        std::sqrt(4.0f), std::sqrt(5.0f), std::sqrt(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Abs) {
    // Modify A to contain negative values (on GPU)
    float host_neg_data[6] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f};
    cudaMemcpy(A.bytes(), host_neg_data, 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor result = abs(A);
    const float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    compareWithExpected(result, expected);
}  
 

TEST_F(TestCUDAFunctions, Sin) {
    Tensor result = sin(A);
    const float expected[6] = {
        std::sin(1.0f), std::sin(2.0f), std::sin(3.0f),
        std::sin(4.0f), std::sin(5.0f), std::sin(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Cos) {
    Tensor result = cos(A);
    const float expected[6] = {
        std::cos(1.0f), std::cos(2.0f), std::cos(3.0f),
        std::cos(4.0f), std::cos(5.0f), std::cos(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Tan) {
    Tensor result = tan(A);
    const float expected[6] = {
        std::tan(1.0f), std::tan(2.0f), std::tan(3.0f),
        std::tan(4.0f), std::tan(5.0f), std::tan(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Sinh) {
    Tensor result = sinh(A);
    const float expected[6] = {
        std::sinh(1.0f), std::sinh(2.0f), std::sinh(3.0f),
        std::sinh(4.0f), std::sinh(5.0f), std::sinh(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Cosh) {
    Tensor result = cosh(A);
    const float expected[6] = {
        std::cosh(1.0f), std::cosh(2.0f), std::cosh(3.0f),
        std::cosh(4.0f), std::cosh(5.0f), std::cosh(6.0f)
    };
    compareWithExpected(result, expected);
}

TEST_F(TestCUDAFunctions, Tanh) {
    Tensor result = tanh(A);
    const float expected[6] = {
        std::tanh(1.0f), std::tanh(2.0f), std::tanh(3.0f),
        std::tanh(4.0f), std::tanh(5.0f), std::tanh(6.0f)
    };
    compareWithExpected(result, expected);
}
#endif