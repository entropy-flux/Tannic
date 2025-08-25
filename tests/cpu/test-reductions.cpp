#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cstring>
 
#include "tensor.hpp"   
#include "reductions.hpp"

using namespace tannic;

TEST(TestReductions, Test1D) {
    Tensor X = {3.0f, 5.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f};
    Tensor tmax = argmax(X);
    Tensor tmin = argmin(X); 
    ASSERT_EQ(tmax[0], 5);
    ASSERT_EQ(tmin[0], 3);
}

TEST(TestReductions, Test2D) {
    Tensor X = {
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    };

    Tensor Zmax = argmax(X);
    Tensor Zmin = argmin(X); 

    ASSERT_EQ(Zmax[0], 1);
    ASSERT_EQ(Zmax[1], 1);
    ASSERT_EQ(Zmax[2], 2);

    ASSERT_EQ(Zmin[0], 3);
    ASSERT_EQ(Zmin[1], 2);
    ASSERT_EQ(Zmin[2], 1);
}
 
TEST(TestReductions, Test2D_Axis0) {
    Tensor X = {
        {3.0f, 5.0f, 4.0f, 1.0f},
        {5.0f, 9.0f, 2.0f, 7.0f},
        {6.0f, 2.0f, 8.0f, 4.0f}
    };
 
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

TEST(TestReductions, TestSum2D_Keepdim) {
    Tensor X = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
 
    Tensor result = sum(X, 1, true);  
    ASSERT_EQ(result.shape(), Shape({2, 1}));  
    ASSERT_EQ(result[0][0], 6.0f);   
    ASSERT_EQ(result[1][0], 15.0f);  
}

TEST(TestReductions, TestMean2D) {
    Tensor X = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    Tensor result = mean(X, 0); 
    
    ASSERT_EQ(result.shape(), Shape({3}));   
    ASSERT_EQ(result[0], 2.5f);  
    ASSERT_EQ(result[1], 3.5f);  
    ASSERT_EQ(result[2], 4.5f);  
} 

TEST(TestReductions, Test1D_MeanAndSum) { 
    Tensor X = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
 
    Tensor tsum = sum(X); 
    ASSERT_EQ(tsum[0], 150.0f);   
 
    Tensor tmean = mean(X); 
    ASSERT_EQ(tmean[0], 30.0f);   
}

TEST(TestReductions, Test1D_MeanAndSum_Keepdim) { 
    Tensor X = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    Tensor tsum = sum(X, 0, true);
    ASSERT_EQ(tsum.shape(), Shape({1}));
    ASSERT_EQ(tsum[0], 150.0f);

    Tensor tmean = mean(X, 0, true); // keepdim = true 
    ASSERT_EQ(tmean.shape(), Shape({1}));
    ASSERT_EQ(tmean[0], 30.0f);
}

TEST(TestReductions, Test2D_AxisMinus1_Keepdim) {
    Tensor X = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    };

    Tensor tsum = sum(X, -1, true);   // keepdim=true
    Tensor tmean = mean(X, -1, true); // keepdim=true 

    ASSERT_EQ(tsum.shape(), Shape({2,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,1}));

    ASSERT_EQ(tsum[0][0], 10.0f);
    ASSERT_EQ(tsum[1][0], 26.0f);

    ASSERT_EQ(tmean[0][0], 2.5f);
    ASSERT_EQ(tmean[1][0], 6.5f);
}

TEST(TestReductions, Test3D_AxisMinus1_Keepdim) {
    Tensor X = {
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
    };

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


TEST(TestReductions, TestRMSIssued) {
    Tensor X = {
        { {1.0f, 2.0f, 3.0f},  {4.0f, 5.0f, 6.0f} },
        { {-1.0f, -2.0f, -3.0f}, {0.5f, 1.0f, 1.5f} }
    };

    Tensor Y = mean(X*X, -1);
    float* data = reinterpret_cast<float*>(Y.bytes()); 

    EXPECT_NEAR(data[0], 4.6667f, 1e-3);
    EXPECT_NEAR(data[1], 25.6667f, 1e-3);
    EXPECT_NEAR(data[2], 4.6667f, 1e-3);
    EXPECT_NEAR(data[3], 1.1667f, 1e-3);
}

/*
TEST(TestReductions, Test1D) {
    Tensor X(float32, {7}); X.initialize(Device());
    X[0] = 3; X[1] = 5; X[2] = 4; X[3] = 1; X[4] = 5; X[5] = 9; X[6] = 2;
    Tensor tmax = argmax(X);
    Tensor tmin = argmin(X); 
    ASSERT_EQ(tmax[0], 5);
    ASSERT_EQ(tmin[0], 3);
}

TEST(TestReductions, Test2D) {
    Tensor X(float32, {3, 4});  X.initialize(Device());
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;

    Tensor Zmax = argmax(X);
    Tensor Zmin = argmin(X); 

    ASSERT_EQ(Zmax[0], 1);
    ASSERT_EQ(Zmax[1], 1);
    ASSERT_EQ(Zmax[2], 2);

    ASSERT_EQ(Zmin[0], 3);
    ASSERT_EQ(Zmin[1], 2);
    ASSERT_EQ(Zmin[2], 1);
}
 
TEST(TestReductions, Test2D_Axis0) {
    Tensor X(float32, {3, 4});  X.initialize(Device());
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;
 
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

TEST(TestReductions, TestSum2D_Keepdim) {
    Tensor X(float32, {2, 3}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3;
    X[1][0] = 4; X[1][1] = 5; X[1][2] = 6;
 
    Tensor result = sum(X, 1, true);  
    ASSERT_EQ(result.shape(), Shape({2, 1}));  
    ASSERT_EQ(result[0][0], 6.0f);   
    ASSERT_EQ(result[0][1], 15.0f);  
}

TEST(TestReductions, TestMean2D) {
    Tensor X(float32, {2, 3}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3;
    X[1][0] = 4; X[1][1] = 5; X[1][2] = 6;

    Tensor result = mean(X, 0); 
    
    ASSERT_EQ(result.shape(), Shape({3}));   
    ASSERT_EQ(result[0], 2.5f);  
    ASSERT_EQ(result[1], 3.5f);  
    ASSERT_EQ(result[2], 4.5f);  
} 

TEST(TestReductions, Test1D_MeanAndSum) { 
    Tensor X(float32, {5}); X.initialize(Device());
    X[0] = 10; X[1] = 20; X[2] = 30; X[3] = 40; X[4] = 50;
 
    Tensor tsum = sum(X); 
    ASSERT_EQ(tsum[0], 150.0f);   
 
    Tensor tmean = mean(X); 
    ASSERT_EQ(tmean[0], 30.0f);   
}

TEST(TestReductions, Test1D_MeanAndSum_Keepdim) { 
    Tensor X(float32, {5}); X.initialize(Device());
    X[0] = 10; X[1] = 20; X[2] = 30; X[3] = 40; X[4] = 50;

    Tensor tsum = sum(X, 0, true);
    ASSERT_EQ(tsum.shape(), Shape({1}));
    ASSERT_EQ(tsum[0], 150.0f);

    Tensor tmean = mean(X, 0, true); // keepdim = true 
    ASSERT_EQ(tmean.shape(), Shape({1}));
    ASSERT_EQ(tmean[0], 30.0f);
}

TEST(TestReductions, Test2D_AxisMinus1_Keepdim) {
    Tensor X(float32, {2, 4}); X.initialize(Device());
    X[0][0] = 1; X[0][1] = 2; X[0][2] = 3; X[0][3] = 4;
    X[1][0] = 5; X[1][1] = 6; X[1][2] = 7; X[1][3] = 8;

    Tensor tsum = sum(X, -1, true);   // keepdim=true
    Tensor tmean = mean(X, -1, true); // keepdim=true 

    ASSERT_EQ(tsum.shape(), Shape({2,1}));
    ASSERT_EQ(tmean.shape(), Shape({2,1}));

    ASSERT_EQ(tsum[0][0], 10.0f);
    ASSERT_EQ(tsum[0][1], 26.0f);

    ASSERT_EQ(tmean[0][0], 2.5f);
    ASSERT_EQ(tmean[0][1], 6.5f);
}

TEST(TestReductions, Test3D_AxisMinus1_Keepdim) {
    Tensor X(float32, {2, 3, 4}); X.initialize(Device());
    
    X[0][0][0] = 1;  X[0][0][1] = 2;  X[0][0][2] = 3;  X[0][0][3] = 4;
    X[0][1][0] = 5;  X[0][1][1] = 6;  X[0][1][2] = 7;  X[0][1][3] = 8;
    X[0][2][0] = 9;  X[0][2][1] = 10; X[0][2][2] = 11; X[0][2][3] = 12;
    
    X[1][0][0] = 13; X[1][0][1] = 14; X[1][0][2] = 15; X[1][0][3] = 16;
    X[1][1][0] = 17; X[1][1][1] = 18; X[1][1][2] = 19; X[1][1][3] = 20;
    X[1][2][0] = 21; X[1][2][1] = 22; X[1][2][2] = 23; X[1][2][3] = 24;

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

 
*/