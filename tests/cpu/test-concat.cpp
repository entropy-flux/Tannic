#include <gtest/gtest.h>

#include "Tensor.hpp"
#include "Transformations.hpp"

using namespace tannic;

TEST(TestConcat, ConcatDim0) {
    Tensor A(float32, {2, 2}); A.initialize();
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;

    Tensor B(float32, {3, 2}); B.initialize();
    B[0][0] = 5; B[0][1] = 6;
    B[1][0] = 7; B[1][1] = 8;
    B[2][0] = 9; B[2][1] = 10;

    Tensor Y = concatenate(A, B, 0);

    ASSERT_EQ(Y.shape()[0], 5);
    ASSERT_EQ(Y.shape()[1], 2);
 
    ASSERT_EQ(Y[0][0], 1);
    ASSERT_EQ(Y[0][1], 2);
    ASSERT_EQ(Y[1][0], 3);
    ASSERT_EQ(Y[1][1], 4);
 
    ASSERT_EQ(Y[2][0], 5);
    ASSERT_EQ(Y[2][1], 6);
    ASSERT_EQ(Y[3][0], 7);
    ASSERT_EQ(Y[3][1], 8);
    ASSERT_EQ(Y[4][0], 9);
    ASSERT_EQ(Y[4][1], 10);
}

TEST(TestConcat, ConcatDim1) {
    Tensor A(float32, {2, 2}); A.initialize();
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;

    Tensor B(float32, {2, 3}); B.initialize();
    B[0][0] = 5; B[0][1] = 6; B[0][2] = 7;
    B[1][0] = 8; B[1][1] = 9; B[1][2] = 10;

    Tensor Y = concatenate(A, B, 1);

    ASSERT_EQ(Y.shape()[0], 2);
    ASSERT_EQ(Y.shape()[1], 5);

    ASSERT_EQ(Y[0][0], 1);
    ASSERT_EQ(Y[0][1], 2);
    
    ASSERT_EQ(Y[0][2], 5);
    ASSERT_EQ(Y[0][3], 6);
    ASSERT_EQ(Y[0][4], 7);

    ASSERT_EQ(Y[1][0], 3);
    ASSERT_EQ(Y[1][1], 4); 

    ASSERT_EQ(Y[1][2], 8);
    ASSERT_EQ(Y[1][3], 9);
    ASSERT_EQ(Y[1][4], 10);
}
