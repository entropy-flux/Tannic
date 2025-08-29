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

TEST(TestReductionsDVC, Test1D) {
    Tensor X(float32, {7});
    X.initialize({3.0f, 5.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f}, Device());

    Tensor tmax = argmax(X, 0);
    Tensor tmin = argmin(X, 0); 
    ASSERT_EQ(tmax[0], 5);
    ASSERT_EQ(tmin[0], 3);
}

TEST(TestReductionsDVC, Test2D) {
    Tensor X(float32, {3,4});
    X.initialize({
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    }, Device());

    Tensor Zmax = argmax(X, -1);
    Tensor Zmin = argmin(X, -1); 

    ASSERT_EQ(Zmax[0], 1);
    ASSERT_EQ(Zmax[1], 1);
    ASSERT_EQ(Zmax[2], 2);

    ASSERT_EQ(Zmin[0], 3);
    ASSERT_EQ(Zmin[1], 2);
    ASSERT_EQ(Zmin[2], 1);
}
 
TEST(TestReductionsDVC, Test2D_Axis0) {
    Tensor X(float32, {3,4});
    X.initialize({
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    }, Device());
 
    Tensor Zmax = argmax(X, 0);
    Tensor Zmin = argmin(X, 0);
 
    ASSERT_EQ(Zmax[0], 2);   
    ASSERT_EQ(Zmax[1], 1);   
    ASSERT_EQ(Zmax[2], 2);   
    ASSERT_EQ(Zmax[3], 1);   
 
    ASSERT_EQ(Zmin[0], 0);  
    ASSERT_EQ(Zmin[1], 2);    
    ASSERT_EQ(Zmin[2], 1);   
    ASSERT_EQ(Zmin[3], 0); 
} 

TEST(TestReductionsDVC, TestSum2D_Keepdim) {
    Tensor X(float32, {2,3});
    X.initialize({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    }, Device());
 
    Tensor result = sum(X, 1, true);  
    ASSERT_EQ(result.shape(), Shape({2, 1}));  
    ASSERT_EQ(result[0][0], 6.0f);   
    ASSERT_EQ(result[1][0], 15.0f);  
}

TEST(TestReductionsDVC, TestMean2D) {
    Tensor X(float32, {2,3});
    X.initialize({
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    }, Device());

    Tensor result = mean(X, 0); 
    
    ASSERT_EQ(result.shape(), Shape({3}));   
    ASSERT_EQ(result[0], 2.5f);  
    ASSERT_EQ(result[1], 3.5f);  
    ASSERT_EQ(result[2], 4.5f);  
} 

TEST(TestReductionsDVC, Test1D_MeanAndSum) { 
    Tensor X(float32, {5});
    X.initialize({10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, Device());
 
    Tensor tsum = sum(X, -1); 
    ASSERT_EQ(tsum[0], 150.0f);   
 
    Tensor tmean = mean(X, -1); 
    ASSERT_EQ(tmean[0], 30.0f);   
}

TEST(TestReductionsDVC, Test1D_MeanAndSum_Keepdim) { 
    Tensor X(float32, {5});
    X.initialize({10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, Device());

    Tensor tsum = sum(X, 0, true);
    ASSERT_EQ(tsum.shape(), Shape({1}));
    ASSERT_EQ(tsum[0], 150.0f);

    Tensor tmean = mean(X, 0, true); // keepdim = true 
    ASSERT_EQ(tmean.shape(), Shape({1}));
    ASSERT_EQ(tmean[0], 30.0f);
}

TEST(TestReductionsDVC, Test2D_AxisMinus1_Keepdim) {
    Tensor X(float32, {2,4});
    X.initialize({
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    }, Device());

    Tensor tsum = sum(X, -1, true);   // keepdim=true
    Tensor tmean = mean(X, -1, true); // keepdim=true 

    ASSERT_EQ(tsum.shape(), Shape({2,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,1}));

    ASSERT_EQ(tsum[0][0], 10.0f);
    ASSERT_EQ(tsum[1][0], 26.0f);

    ASSERT_EQ(tmean[0][0], 2.5f);
    ASSERT_EQ(tmean[1][0], 6.5f);
}

TEST(TestReductionsDVC, Test3D_AxisMinus1_Keepdim) {
    Tensor X(float32, {2,3,4});
    X.initialize({
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

    Tensor tsum = sum(X, -1, true);
    Tensor tmean = mean(X, -1, true);
 
    ASSERT_EQ(tsum.shape(), Shape({2,3,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,3,1}));

    ASSERT_EQ(tsum[0][0][0], 10.0f);
    ASSERT_EQ(tsum[0][1][0], 26.0f);
    ASSERT_EQ(tsum[0][2][0], 42.0f);
    ASSERT_EQ(tsum[1][0][0], 58.0f);
    ASSERT_EQ(tsum[1][1][0], 74.0f);
    ASSERT_EQ(tsum[1][2][0], 90.0f);

    ASSERT_EQ(tmean[0][0][0], 2.5f);
    ASSERT_EQ(tmean[0][1][0], 6.5f);
    ASSERT_EQ(tmean[0][2][0], 10.5f);
    ASSERT_EQ(tmean[1][0][0], 14.5f);
    ASSERT_EQ(tmean[1][1][0], 18.5f);
    ASSERT_EQ(tmean[1][2][0], 22.5f); 
}  

#endif