#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "Tensor.hpp"
#include "Complex.hpp"

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
    Tensor Z = complex(X);  
 
    float* device_data = reinterpret_cast<float*>(Z.bytes()); 
    float host_data[4];

    cudaError_t err = cudaMemcpy(host_data, device_data, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
 
    ASSERT_FLOAT_EQ(host_data[0], 1);  
    ASSERT_FLOAT_EQ(host_data[1], 0);    
    ASSERT_FLOAT_EQ(host_data[2], 2);  
    ASSERT_FLOAT_EQ(host_data[3], 3);  

    Tensor Y = real(Z);  
    ASSERT_EQ(X[0][0], 1);
    ASSERT_EQ(X[0][1], 0); 
    ASSERT_EQ(X[1][0], 2); 
    ASSERT_EQ(X[1][1], 3); 
}
#endif