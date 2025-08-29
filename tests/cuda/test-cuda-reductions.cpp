#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cstring>
#include <cuda_runtime.h>  
 
#include "tensor.hpp"   
#include "reductions.hpp"
#include "comparisons.hpp"

using namespace tannic;

TEST(TestReductionsDVC, Test1DDVC) {
    Tensor X(float32, {7}); X.initialize({3.0f, 5.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f}, Device());
    Tensor tmax = argmax(X);
    Tensor tmin = argmin(X); 
    ASSERT_EQ(tmax[0], 5);  // Index of max value (9.0f)
    ASSERT_EQ(tmin[0], 3);  // Index of min value (1.0f)
}

TEST(TestReductionsDVC, Test2DDVC) {
    // Fixed: Changed shape from {3,5} to {3,4} to match data
    Tensor X(float32, {3,4}); X.initialize({
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    }, Device());

    Tensor Zmax = argmax(X);  // Should return indices along flattened array
    Tensor Zmin = argmin(X); 

    // These assertions depend on how argmax/argmin work with 2D tensors
    // If they flatten the tensor, then:
    // Flattened: [3,5,4,1,5,9,2,7,6,2,8,4]
    // Max at index 5 (value 9), Min at index 3 (value 1)
    ASSERT_EQ(Zmax[0], 5);
    ASSERT_EQ(Zmin[0], 3);
}

TEST(TestReductionsDVC, Test2D_Axis0DVC) {
    // Fixed: Changed shape from {3,5} to {3,4}
    Tensor X(float32, {3,4}); X.initialize({
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    }, Device());
 
    Tensor Zmax = argmax(X, 0);  // Along columns (axis 0)
    Tensor Zmin = argmin(X, 0);
 
    // For each column, find row index with max/min value:
    // Col 0: [3,5,6] -> max at row 2, min at row 0
    // Col 1: [5,9,2] -> max at row 1, min at row 2  
    // Col 2: [4,2,8] -> max at row 2, min at row 1
    // Col 3: [1,7,4] -> max at row 1, min at row 0
    ASSERT_EQ(Zmax[0], 2);   
    ASSERT_EQ(Zmax[1], 1);   
    ASSERT_EQ(Zmax[2], 2);   
    ASSERT_EQ(Zmax[3], 1);   
 
    ASSERT_EQ(Zmin[0], 0);  
    ASSERT_EQ(Zmin[1], 2);    
    ASSERT_EQ(Zmin[2], 1);   
    ASSERT_EQ(Zmin[3], 0); 
}

TEST(TestReductionsDVC, TestSum2D_KeepdimDVC) {
    Tensor X(float32,{2,3}); X.initialize({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    }, Device());
 
    Tensor result = sum(X, 1, true);  // Sum along axis 1 (columns), keepdim
    ASSERT_EQ(result.shape(), Shape({2, 1}));  
    ASSERT_EQ(result[0][0], 6.0f);   // 1+2+3 = 6
    ASSERT_EQ(result[1][0], 15.0f);  // 4+5+6 = 15
}

TEST(TestReductionsDVC, TestMean2DDVC) {
    Tensor X(float32, {2,3}); X.initialize({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    }, Device());

    Tensor result = mean(X, 0);  // Mean along axis 0 (rows)
    
    ASSERT_EQ(result.shape(), Shape({3}));   
    ASSERT_EQ(result[0], 2.5f);  // (1+4)/2 = 2.5
    ASSERT_EQ(result[1], 3.5f);  // (2+5)/2 = 3.5
    ASSERT_EQ(result[2], 4.5f);  // (3+6)/2 = 4.5
}

TEST(TestReductionsDVC, Test1D_MeanAndSumDVC) { 
    Tensor X(float32, {5}); X.initialize({10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, Device());
 
    Tensor tsum = sum(X); 
    ASSERT_EQ(tsum[0], 150.0f);   // 10+20+30+40+50 = 150
 
    Tensor tmean = mean(X); 
    ASSERT_EQ(tmean[0], 30.0f);   // 150/5 = 30
}

// Fixed test name: Added "DVC" to test suite name
TEST(TestReductionsDVC, Test1D_MeanAndSum_KeepdimDVC) { 
    Tensor X(float32, {5}); X.initialize({10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, Device());

    Tensor tsum = sum(X, 0, true);
    ASSERT_EQ(tsum.shape(), Shape({1}));
    ASSERT_EQ(tsum[0], 150.0f);

    Tensor tmean = mean(X, 0, true);
    ASSERT_EQ(tmean.shape(), Shape({1}));
    ASSERT_EQ(tmean[0], 30.0f);
}

TEST(TestReductionsDVC, Test2D_AxisMinus1_KeepdimDVC) {
    Tensor X(float32, {2,4}); X.initialize({
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    }, Device());

    Tensor tsum = sum(X, -1, true);   // Sum along last axis (columns)
    Tensor tmean = mean(X, -1, true);

    ASSERT_EQ(tsum.shape(), Shape({2,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,1}));

    ASSERT_EQ(tsum[0][0], 10.0f);  // 1+2+3+4 = 10
    ASSERT_EQ(tsum[1][0], 26.0f);  // 5+6+7+8 = 26

    ASSERT_EQ(tmean[0][0], 2.5f);  // 10/4 = 2.5
    ASSERT_EQ(tmean[1][0], 6.5f);  // 26/4 = 6.5
}

TEST(TestReductionsDVC, Test3D_AxisMinus1_KeepdimDVC) {
    Tensor X(float32, {2,3,4}); X.initialize({
        {
            {1.0f,  2.0f,  3.0f,  4.0f},
            {5.0f,  6.0f,  7.0f,  8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f}
        },
        {
            {13.0f, 14.0f, 15.0f, 16.0f},
            {17.0f, 18.0f, 19.0f, 20.0f},
            {21.0f, 22.0f, 23.0f, 24.0f}
        }
    }, Device());

    Tensor tsum = sum(X, -1, true);   // Sum along last axis (4th dimension)
    Tensor tmean = mean(X, -1, true);
 
    ASSERT_EQ(tsum.shape(), Shape({2,3,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,3,1}));

    // Sums along the last dimension (4 elements each)
    ASSERT_EQ(tsum[0][0][0], 10.0f);  // 1+2+3+4
    ASSERT_EQ(tsum[0][1][0], 26.0f);  // 5+6+7+8
    ASSERT_EQ(tsum[0][2][0], 42.0f);  // 9+10+11+12
    ASSERT_EQ(tsum[1][0][0], 58.0f);  // 13+14+15+16
    ASSERT_EQ(tsum[1][1][0], 74.0f);  // 17+18+19+20
    ASSERT_EQ(tsum[1][2][0], 90.0f);  // 21+22+23+24

    // Means along the last dimension
    ASSERT_EQ(tmean[0][0][0], 2.5f);   // 10/4
    ASSERT_EQ(tmean[0][1][0], 6.5f);   // 26/4
    ASSERT_EQ(tmean[0][2][0], 10.5f);  // 42/4
    ASSERT_EQ(tmean[1][0][0], 14.5f);  // 58/4
    ASSERT_EQ(tmean[1][1][0], 18.5f);  // 74/4
    ASSERT_EQ(tmean[1][2][0], 22.5f);  // 90/4
}

#endif