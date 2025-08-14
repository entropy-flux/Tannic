#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cstring>
#include <cuda_runtime.h>  
 
#include "Tensor.hpp"   
#include "Reductions.hpp"

using namespace tannic;

TEST(TestCUDAReductions, TestCUDA1D) {
    Tensor X(float32, {7}); X.initialize();
    X[0] = 3; X[1] = 5; X[2] = 4; X[3] = 1; X[4] = 5; X[5] = 9; X[6] = 2;
    
    Tensor tmax = argmax(X);
    Tensor tmin = argmin(X);

    int64_t host_imax[1];
    int64_t host_imin[1];

    cudaMemcpy(host_imax, tmax.bytes(), sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_imin, tmin.bytes(), sizeof(int64_t), cudaMemcpyDeviceToHost);

    ASSERT_EQ(host_imax[0], 5);
    ASSERT_EQ(host_imin[0], 3);
}

TEST(TestCUDAReductions, TestCUDA2D) {
    Tensor X(float32, {3, 4});
    X.initialize();
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;

    Tensor Zmax = argmax(X);
    Tensor Zmin = argmin(X);

    int64_t host_zmax[3];
    int64_t host_zmin[3];

    cudaMemcpy(host_zmax, Zmax.bytes(), sizeof(host_zmax), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_zmin, Zmin.bytes(), sizeof(host_zmin), cudaMemcpyDeviceToHost);

    ASSERT_EQ(host_zmax[0], 1);
    ASSERT_EQ(host_zmax[1], 1);
    ASSERT_EQ(host_zmax[2], 2);

    ASSERT_EQ(host_zmin[0], 3);
    ASSERT_EQ(host_zmin[1], 2);
    ASSERT_EQ(host_zmin[2], 1);
}


 
TEST(TestCUDAReductions, TestCUDA2D_Axis0) {
    Tensor X(float32, {3, 4}); X.initialize();
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;
 
    Tensor Zmax = argmax(X, 0);
    Tensor Zmin = argmin(X, 0);

    uint64_t host_zmax[4];
    uint64_t host_zmin[4];

    cudaMemcpy(host_zmax, Zmax.bytes(), sizeof(host_zmax), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_zmin, Zmin.bytes(), sizeof(host_zmin), cudaMemcpyDeviceToHost);

    ASSERT_EQ(host_zmax[0], 2);   
    ASSERT_EQ(host_zmax[1], 1);   
    ASSERT_EQ(host_zmax[2], 2);   
    ASSERT_EQ(host_zmax[3], 1);   
 
    ASSERT_EQ(host_zmin[0], 0);  
    ASSERT_EQ(host_zmin[1], 2);    
    ASSERT_EQ(host_zmin[2], 1);   
    ASSERT_EQ(host_zmin[3], 0); 
}  
 

TEST(TestCUDAReductions, TestCUDASum2D_Keepdim) {
    Tensor X(float32, {2, 3}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3;
    X[1][0] = 4; X[1][1] = 5; X[1][2] = 6;

    Tensor result = sum(X, 1, true);  // keepdim = true

    ASSERT_EQ(result.shape(), Shape({2, 1}));  // shape check

    float host_data[2];
    cudaMemcpy(host_data, result.bytes(), sizeof(host_data), cudaMemcpyDeviceToHost);

    ASSERT_FLOAT_EQ(host_data[0], 6.0f);   // 1+2+3
    ASSERT_FLOAT_EQ(host_data[1], 15.0f);  // 4+5+6
}


TEST(TestCUDAReductions, TestCUDAMean2D) {
    Tensor X(float32, {2, 3}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3;
    X[1][0] = 4; X[1][1] = 5; X[1][2] = 6;

    Tensor result = mean(X, 0);  // reduce along axis 0

    ASSERT_EQ(result.shape(), Shape({3}));  // reduced shape

    float host_data[3];
    cudaMemcpy(host_data, result.bytes(), sizeof(host_data), cudaMemcpyDeviceToHost);

    ASSERT_FLOAT_EQ(host_data[0], 2.5f);  // (1+4)/2
    ASSERT_FLOAT_EQ(host_data[1], 3.5f);  // (2+5)/2
    ASSERT_FLOAT_EQ(host_data[2], 4.5f);  // (3+6)/2
}

#endif