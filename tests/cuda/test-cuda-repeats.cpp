#ifdef CUDA
#include <gtest/gtest.h>
 
#include "tensor.hpp"   
#include "transformations.hpp"

using namespace tannic;

TEST(TestRepeatsCUDA, TestRepsCUDA) {
    Tensor X(float32, {2,2}); X.initialize(Device());
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


TEST(TestRepeatsCUDA, TestRepeatDim1CUDA) {
    Tensor X(float32, {2, 2}); X.initialize(Device());
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


 TEST(TestRepeats, TestRepeatAfterPermute) {
    Tensor X(float32, {2, 2, 3});
    X.initialize(Device());

    
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 3; k++)
                X[i][j][k] = val++;

                
    Tensor P = permute(X, 2,0,1);

    
    Tensor Y = repeat(P, 2, 0);

    ASSERT_EQ(Y.shape()[0], 6);
    ASSERT_EQ(Y.shape()[1], 2);
    ASSERT_EQ(Y.shape()[2], 2);

    
    EXPECT_EQ(Y[0][0][0], 1);
    EXPECT_EQ(Y[0][0][1], 4);
    EXPECT_EQ(Y[0][1][0], 7);
    EXPECT_EQ(Y[0][1][1], 10);

    EXPECT_EQ(Y[1][0][0], 1);
    EXPECT_EQ(Y[1][0][1], 4);
    EXPECT_EQ(Y[1][1][0], 7);
    EXPECT_EQ(Y[1][1][1], 10); 

    EXPECT_EQ(Y[2][0][0], 2);
    EXPECT_EQ(Y[2][0][1], 5);
    EXPECT_EQ(Y[2][1][0], 8);
    EXPECT_EQ(Y[2][1][1], 11);

    EXPECT_EQ(Y[3][0][0], 2);
    EXPECT_EQ(Y[3][0][1], 5);
    EXPECT_EQ(Y[3][1][0], 8);
    EXPECT_EQ(Y[3][1][1], 11);
 
    EXPECT_EQ(Y[4][0][0], 3);
    EXPECT_EQ(Y[4][0][1], 6);
    EXPECT_EQ(Y[4][1][0], 9);
    EXPECT_EQ(Y[4][1][1], 12);

    EXPECT_EQ(Y[5][0][0], 3);
    EXPECT_EQ(Y[5][0][1], 6);
    EXPECT_EQ(Y[5][1][0], 9);
    EXPECT_EQ(Y[5][1][1], 12);
}


#endif