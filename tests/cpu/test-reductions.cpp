#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cstring>
 
#include "Tensor.hpp"   
#include "Reductions.hpp"

using namespace tannic;

TEST(TestReductions, Test1D) {
    Tensor X(float32, {7}); X.initialize();
    X[0] = 3; X[1] = 5; X[2] = 4; X[3] = 1; X[4] = 5; X[5] = 9; X[6] = 2;
    
    Tensor tmax = argmax(X);
    Tensor tmin = argmin(X);
    int64_t* imax = reinterpret_cast<int64_t*>(tmax.bytes());
    int64_t* imin = reinterpret_cast<int64_t*>(tmin.bytes());
    ASSERT_EQ(imax[0], 5);
    ASSERT_EQ(imin[0], 3);
}

TEST(TestReductions, Test2D) {
    Tensor X(float32, {3, 4});
    X.initialize();
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;

    Tensor Zmax = argmax(X);
    Tensor Zmin = argmin(X);
   
    int64_t* zmaxdat = reinterpret_cast<int64_t*>(Zmax.bytes());
    int64_t* zmindat = reinterpret_cast<int64_t*>(Zmin.bytes()); 

    ASSERT_EQ(zmaxdat[0], 1);
    ASSERT_EQ(zmaxdat[1], 1);
    ASSERT_EQ(zmaxdat[2], 2);

    ASSERT_EQ(zmindat[0], 3);
    ASSERT_EQ(zmindat[1], 2);
    ASSERT_EQ(zmindat[2], 1);
}
 
TEST(TestReductions, Test2D_Axis0) {
    Tensor X(float32, {3, 4}); X.initialize();
    X[0][0] = 3; X[0][1] = 5; X[0][2] = 4; X[0][3] = 1;
    X[1][0] = 5; X[1][1] = 9; X[1][2] = 2; X[1][3] = 7;
    X[2][0] = 6; X[2][1] = 2; X[2][2] = 8; X[2][3] = 4;
 
    Tensor Zmax = argmax(X, 0);
    Tensor Zmin = argmin(X, 0);

    std::cout << Zmax;
    std::cout << Zmin;

    uint64_t* zmaxdat = reinterpret_cast<uint64_t*>(Zmax.bytes());
    uint64_t* zmindat = reinterpret_cast<uint64_t*>(Zmin.bytes()); 

    ASSERT_EQ(zmaxdat[0], 2);   
    ASSERT_EQ(zmaxdat[1], 1);   
    ASSERT_EQ(zmaxdat[2], 2);   
    ASSERT_EQ(zmaxdat[3], 1);   
 
    ASSERT_EQ(zmindat[0], 0);  
    ASSERT_EQ(zmindat[1], 2);    
    ASSERT_EQ(zmindat[2], 1);   
    ASSERT_EQ(zmindat[3], 0); 
} 