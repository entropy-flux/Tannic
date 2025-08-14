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

TEST(TestCUDAReductions, TestCUDA1D_MeanAndSum) {
    // Rank-1 tensor
    Tensor X(float32, {5}); X.initialize(Device());
    X[0] = 10; X[1] = 20; X[2] = 30; X[3] = 40; X[4] = 50;
 
    Tensor tsum = sum(X); 
    float host_sum;
    cudaMemcpy(&host_sum, tsum.bytes(), sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_FLOAT_EQ(host_sum, 150.0f);  // 10+20+30+40+50

    // Mean
    Tensor tmean = mean(X); 
    float host_mean;
    cudaMemcpy(&host_mean, tmean.bytes(), sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_FLOAT_EQ(host_mean, 30.0f);  // 150 / 5
}

TEST(TestCUDAReductions, TestCUDA2D_AxisMinus1) { 
    Tensor X(float32, {2, 4}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3; X[0][3] = 4;
    X[1][0] = 5; X[1][1] = 6; X[1][2] = 7; X[1][3] = 8;
 
    Tensor tsum = sum(X, -1, true);  // keepdim=true
    Tensor tmean = mean(X, -1, true); // keepdim=true

    float host_sum[2];
    float host_mean[2];

    cudaMemcpy(host_sum, tsum.bytes(), sizeof(host_sum), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mean, tmean.bytes(), sizeof(host_mean), cudaMemcpyDeviceToHost);
 
    ASSERT_EQ(tsum.shape(), Shape({2, 1}));
    ASSERT_EQ(tmean.shape(), Shape({2, 1}));
 
    ASSERT_FLOAT_EQ(host_sum[0], 10.0f);  // 1+2+3+4
    ASSERT_FLOAT_EQ(host_sum[1], 26.0f);  // 5+6+7+8
 
    ASSERT_FLOAT_EQ(host_mean[0], 2.5f);  // 10 / 4
    ASSERT_FLOAT_EQ(host_mean[1], 6.5f);  // 26 / 4
}

TEST(TestCUDAReductions, TestCUDA3D_AxisMinus1) {
    // 3D tensor: e.g., batch x seq_len x features
    Tensor X(float32, {2, 3, 4}); X.initialize(Device());

    // Fill with example values
    // Batch 0
    X[0][0][0] = 1; X[0][0][1] = 2; X[0][0][2] = 3; X[0][0][3] = 4;
    X[0][1][0] = 5; X[0][1][1] = 6; X[0][1][2] = 7; X[0][1][3] = 8;
    X[0][2][0] = 9; X[0][2][1] = 10; X[0][2][2] = 11; X[0][2][3] = 12;

    // Batch 1
    X[1][0][0] = 13; X[1][0][1] = 14; X[1][0][2] = 15; X[1][0][3] = 16;
    X[1][1][0] = 17; X[1][1][1] = 18; X[1][1][2] = 19; X[1][1][3] = 20;
    X[1][2][0] = 21; X[1][2][1] = 22; X[1][2][2] = 23; X[1][2][3] = 24;

    // Reduce along last axis (features)
    Tensor tsum = sum(X, -1, true);   // keepdim=true
    Tensor tmean = mean(X, -1, true); // keepdim=true

    float host_sum[6];   // 2 batches * 3 rows = 6 sums
    float host_mean[6];

    cudaMemcpy(host_sum, tsum.bytes(), sizeof(host_sum), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mean, tmean.bytes(), sizeof(host_mean), cudaMemcpyDeviceToHost);

    // Check shapes
    ASSERT_EQ(tsum.shape(), Shape({2, 3, 1}));
    ASSERT_EQ(tmean.shape(), Shape({2, 3, 1}));

    // Sum: row-wise per batch
    ASSERT_FLOAT_EQ(host_sum[0], 10.0f);   // 1+2+3+4
    ASSERT_FLOAT_EQ(host_sum[1], 26.0f);   // 5+6+7+8
    ASSERT_FLOAT_EQ(host_sum[2], 42.0f);   // 9+10+11+12
    ASSERT_FLOAT_EQ(host_sum[3], 58.0f);   // 13+14+15+16
    ASSERT_FLOAT_EQ(host_sum[4], 74.0f);   // 17+18+19+20
    ASSERT_FLOAT_EQ(host_sum[5], 90.0f);   // 21+22+23+24

    // Mean: row-wise per batch
    ASSERT_FLOAT_EQ(host_mean[0], 2.5f);  // 10 / 4
    ASSERT_FLOAT_EQ(host_mean[1], 6.5f);  // 26 / 4
    ASSERT_FLOAT_EQ(host_mean[2], 10.5f); // 42 / 4
    ASSERT_FLOAT_EQ(host_mean[3], 14.5f); // 58 / 4
    ASSERT_FLOAT_EQ(host_mean[4], 18.5f); // 74 / 4
    ASSERT_FLOAT_EQ(host_mean[5], 22.5f); // 90 / 4
}


#endif