#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>  
#include "Tensor.hpp" 

using namespace tannic;

class TestCUDAOperations : public ::testing::Test {
protected:
    Tensor A;
    Tensor B; 

    TestCUDAOperations() 
    :   A(float32, Shape(2, 1, 3))
    ,   B(float32, Shape(1, 4, 3))
    {
        A.initialize(Device(0));  
        B.initialize(Device(0)); 
    }

    void SetUp() override { 
        float A_data[6] = {
            1.0f, 2.0f, 3.0f, 
            4.0f, 5.0f, 6.0f
        };
        
        float B_data[12] =  
        { 
            0.0f,  10.0f, 20.0f,  
            30.0f, 40.0f, 50.0f,  
            60.0f, 70.0f, 80.0f, 
            90.0f, 100.0f, 110.0f 
        };

        cudaMemcpy(A.bytes(), A_data, 6 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B.bytes(), B_data, 12 * sizeof(float), cudaMemcpyHostToDevice); 
    }
 
    void compareWithExpectedUnary(const Tensor& result, const float expected[6]) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float cpu_data[6];
         
        cudaMemcpy(cpu_data, gpu_data, 6 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(cpu_data[i], expected[i]);
        }
    }

    void compareWithExpectedBinary(const Tensor& result, const float expected[24]) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float cpu_data[24];
         
        cudaMemcpy(cpu_data, gpu_data, 24 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 24; ++i) {
            EXPECT_FLOAT_EQ(cpu_data[i], expected[i]);
        }
    }
};

TEST_F(TestCUDAOperations, Neg) {
    Tensor result = -A;
    const float expected[6] = {
        -1.0f, -2.0f, -3.0f,
        -4.0f, -5.0f, -6.0f
    };
    compareWithExpectedUnary(result, expected);
}

TEST_F(TestCUDAOperations, Addition) {
    Tensor result = A + B;
    const float expected[24] = { 
        1.0f, 12.0f, 23.0f,
        31.0f, 42.0f, 53.0f,
        61.0f, 72.0f, 83.0f,
        91.0f, 102.0f, 113.0f,
         
        4.0f, 15.0f, 26.0f,
        34.0f, 45.0f, 56.0f,
        64.0f, 75.0f, 86.0f,
        94.0f, 105.0f, 116.0f
    };
    compareWithExpectedBinary(result, expected);
}

TEST_F(TestCUDAOperations, Multiplication) {
    Tensor result = A * B;
    const float expected[24] = { 
        0.0f, 20.0f, 60.0f,
        30.0f, 80.0f, 150.0f,
        60.0f, 140.0f, 240.0f,
        90.0f, 200.0f, 330.0f,
         
        0.0f, 50.0f, 120.0f,
        120.0f, 200.0f, 300.0f,
        240.0f, 350.0f, 480.0f,
        360.0f, 500.0f, 660.0f
    };
    compareWithExpectedBinary(result, expected);
}

TEST_F(TestCUDAOperations, Subtraction) {
    Tensor result = A - B;
    const float expected[24] = {  
        1.0f, -8.0f, -17.0f,
        -29.0f, -38.0f, -47.0f,
        -59.0f, -68.0f, -77.0f,
        -89.0f, -98.0f, -107.0f,
         
        4.0f, -5.0f, -14.0f,
        -26.0f, -35.0f, -44.0f,
        -56.0f, -65.0f, -74.0f,
        -86.0f, -95.0f, -104.0f
    };
    compareWithExpectedBinary(result, expected);
}

TEST_F(TestCUDAOperations, Complex_CUDA) { 
    Tensor X(float32, {2,2});
    X.initialize(Device());   
    X[0, 0] = 1;
    X[0, 1] = 6;
    X[1, 0] = 2;
    X[1, 1] = 3;
    X = complex(X);  
    
    
    Tensor Y(float32, {2,2});
    Y.initialize(Device()); 
    
    Y[0, 0] = 2;
    Y[0, 1] = 1;
    Y[1, 0] = 1.5;
    Y[1, 1] = 3.14;
    Y = complex(Y);
    
    Tensor Z = real(X * Y);

    float Z_host[4];
 
    cudaMemcpy(
        Z_host,
        reinterpret_cast<float*>(Z.bytes()),
        4 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    ASSERT_NEAR(Z_host[0], -4.00, 0.001);
    ASSERT_NEAR(Z_host[1], 13.00, 0.001);
    ASSERT_NEAR(Z_host[2], -6.42, 0.001);
    ASSERT_NEAR(Z_host[3], 10.78, 0.001);
}


#endif