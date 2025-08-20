#include <gtest/gtest.h>
 
#include "Tensor.hpp"   
#include "Transformations.hpp"
#include "Views.hpp"

using namespace tannic;

TEST(TestRepeats, TestReps) {
    Tensor X(float32, {2,2}); X.initialize();
    X[0][0] = 1;
    X[0][1] = 2;
    X[1][0] = 3;
    X[1][1] = 4;

    Tensor Y = repeat(X, 2, 0);
 
    ASSERT_EQ(Y.shape()[0], 4);
    ASSERT_EQ(Y.shape()[1], 2);
 
    ASSERT_EQ(Y[0][0], 1);
    ASSERT_EQ(Y[0][1], 2);
    ASSERT_EQ(Y[1][0], 1);
    ASSERT_EQ(Y[1][1], 2);
    ASSERT_EQ(Y[2][0], 3);
    ASSERT_EQ(Y[2][1], 4);
    ASSERT_EQ(Y[3][0], 3);
    ASSERT_EQ(Y[3][1], 4); 
}


TEST(TestRepeats, TestRepeatDim1) {
    Tensor X(float32, {2, 2}); X.initialize();
    X[0][0] = 1;
    X[0][1] = 2;
    X[1][0] = 3;
    X[1][1] = 4;
 
    Tensor Y = repeat(X, 2, 1);
 
    ASSERT_EQ(Y.shape()[0], 2);
    ASSERT_EQ(Y.shape()[1], 4);
 
    ASSERT_EQ(Y[0][0], 1);
    ASSERT_EQ(Y[0][1], 1);
    ASSERT_EQ(Y[0][2], 2);
    ASSERT_EQ(Y[0][3], 2);
    ASSERT_EQ(Y[1][0], 3);
    ASSERT_EQ(Y[1][1], 3);
    ASSERT_EQ(Y[1][2], 4);
    ASSERT_EQ(Y[1][3], 4);
}


TEST(TestPermute, TestPermute234to423) {
    Tensor X(float32, {2,3,4});
    X.initialize();
 
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;

    Tensor Y = permute(X, 2,0,1);   
 
    ASSERT_EQ(Y.shape()[0], 4);
    ASSERT_EQ(Y.shape()[1], 2);
    ASSERT_EQ(Y.shape()[2], 3);
 
    int expected;
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 3; k++) { 
                expected = 1 + j*12 + k*4 + i;  
                EXPECT_EQ(Y[i][j][k], expected)
                    << "Mismatch at Y[" << i << "][" << j << "][" << k << "]";
            }
        }
    }
}