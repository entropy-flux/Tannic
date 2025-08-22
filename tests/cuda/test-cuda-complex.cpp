#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "tensor.hpp"
#include "complex.hpp" 

using namespace tannic;

class CUDAComplexTests : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(CUDAComplexTests, CUDAComplexificationAndRealification) {
    Tensor X(float32, {2,2});  X.initialize(Device());   
    X[0][0] = 1;
    X[0][1] = 0;
    X[1][0] = 2;
    X[1][1] = 3;   
    Tensor Z = complexify(X);  
 
    float* device_data = reinterpret_cast<float*>(Z.bytes()); 
    float host_data[4];

    cudaError_t err = cudaMemcpy(host_data, device_data, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
 
    ASSERT_FLOAT_EQ(host_data[0], 1);  
    ASSERT_FLOAT_EQ(host_data[1], 0);    
    ASSERT_FLOAT_EQ(host_data[2], 2);  
    ASSERT_FLOAT_EQ(host_data[3], 3);  

    Tensor Y = realify(Z);  
    ASSERT_EQ(X[0][0], 1);
    ASSERT_EQ(X[0][1], 0); 
    ASSERT_EQ(X[1][0], 2); 
    ASSERT_EQ(X[1][1], 3); 
}   

TEST_F(CUDAComplexTests, CUDAPolarConversionManual) {
    // Magnitude tensor X (CUDA device)
    Tensor X(float32, {2, 2});
    X.initialize(Device()); 

    X[0][0] = 1.0f;  X[0][1] = 6.0f;
    X[1][0] = 2.0f;  X[1][1] = 3.0f;

    // Phase tensor Y (CUDA device)
    Tensor Y(float32, {2, 2});
    Y.initialize(Device());

    Y[0][0] = 2.0f;  Y[0][1] = 1.0f;
    Y[1][0] = 1.5f;  Y[1][1] = 3.14f;

    // Complex tensor from polar coordinates
    Tensor Z = polar(X, Y);

    // Allocate host buffer to copy back complex data (4 complex numbers * 2 floats each)
    float host_data[8]; // real0, imag0, real1, imag1, ...

    // Copy from device to host
    float* device_data = reinterpret_cast<float*>(Z.bytes());
    cudaError_t err = cudaMemcpy(host_data, device_data, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);

    // Expected values from your PyTorch example
    const float expected_real[4] = {-0.4161f, 3.2418f, 0.1415f, -3.0000f};
    const float expected_imag[4] = {0.9093f, 5.0488f, 1.9950f, 0.0047f};

    // Verify each complex element manually
    for (int i = 0; i < 4; ++i) {
        float real_val = host_data[2 * i];
        float imag_val = host_data[2 * i + 1];

        ASSERT_NEAR(real_val, expected_real[i], 1e-3f) << "Real part mismatch at index " << i;
        ASSERT_NEAR(imag_val, expected_imag[i], 1e-3f) << "Imag part mismatch at index " << i;
    }
}

#endif


/*

import torch

# Create magnitude tensor X
X = torch.tensor([[1.0, 6.0],
                  [2.0, 3.0]])

# Create phase tensor Y (in radians)
Y = torch.tensor([[2.0, 1.0],
                  [1.5, 3.14]])

# Create complex tensor Z from polar coordinates
Z = torch.polar(X, Y)

print("Z real part:\n", Z.real)
print("Z imag part:\n", Z.imag)
 
Z real part:
 tensor([[-0.4161,  3.2418],
        [ 0.1415, -3.0000]])
Z imag part:
 tensor([[9.0930e-01, 5.0488e+00],
        [1.9950e+00, 4.7776e-03]])


*/