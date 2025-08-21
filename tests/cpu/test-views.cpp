#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Views.hpp"
#include "Transformations.hpp"

using namespace tannic;

TEST(TestTensorView, TestBasicView) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;
 
    Tensor Y = X.view(3, 2);
 
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 2);
 
    int expected = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_EQ(Y[i][j], expected)
                << "Mismatch at Y[" << i << "][" << j << "]";
            expected++;
        }
    }
 
    Y[0][0] = 100;
    EXPECT_EQ(X[0][0], 100);
}

TEST(TestTensorReshape, TestBasicReshape) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;
 
    Tensor Y = reshape(X, 3, 2);
 
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 2);
 
    int expected = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_EQ(Y[i][j], expected)
                << "Mismatch at Y[" << i << "][" << j << "]";
            expected++;
        }
    }
 
    Y[0][0] = 100;
    EXPECT_EQ(X[0][0], 1);
}

TEST(TestTensorExpand, TestBasicExpand) {
    Tensor X(float32, {1, 3}); 
    X.initialize();
 
    X[0][0] = 1;
    X[0][1] = 2;
    X[0][2] = 3;
 
    Tensor Y = expand(X, 4, 3);

    ASSERT_EQ(Y.shape()[0], 4);
    ASSERT_EQ(Y.shape()[1], 3);
 
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(Y[i][0], 1) << "Mismatch at Y[" << i << "][0]";
        EXPECT_EQ(Y[i][1], 2) << "Mismatch at Y[" << i << "][1]";
        EXPECT_EQ(Y[i][2], 3) << "Mismatch at Y[" << i << "][2]";
    }
  
    Y[2][1] = 42;
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(Y[i][1], 42) << "Mismatch at Y[" << i << "][1]";
    }
}