#include <gtest/gtest.h>
#include "tensor.hpp"
#include "views.hpp"
#include "transformations.hpp"

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

TEST(TestTensorView, TestViewInferMiddleDimension) {
    Tensor X(float32, {2, 3, 4}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;
 
    Tensor Y = X.view(2, -1, 2);

    ASSERT_EQ(Y.shape().rank(), 3);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 6);
    EXPECT_EQ(Y.shape()[2], 2);
 
    EXPECT_EQ(Y.shape()[0] * Y.shape()[1] * Y.shape()[2], 24);
 
    int expected = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 6; j++)
            for (size_t k = 0; k < 2; k++)
                EXPECT_EQ(Y[i][j][k], expected++);
}

TEST(TestTensorView, TestViewInferFirstDimension) {
    Tensor X(float32, {2, 3, 4}); 
 
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;
 
    Tensor Y = X.view(-1, 3, 4);

    ASSERT_EQ(Y.shape().rank(), 3);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 3);
    EXPECT_EQ(Y.shape()[2], 4);

    int expected = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                EXPECT_EQ(Y[i][j][k], expected++);
}

TEST(TestTensorView, TestViewInferLastDimension) {
    Tensor X(float32, {2, 3, 4});  
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;

    Tensor Y = X.view(4, -1);

    ASSERT_EQ(Y.shape().rank(), 2);
    EXPECT_EQ(Y.shape()[0], 4);
    EXPECT_EQ(Y.shape()[1], 6);

    int expected = 1;
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(Y[i][j], expected++);
}

TEST(TestTensorView, TestViewInferInvalidMultiple) {
    Tensor X(float32, {2, 3, 4}); 
    X.initialize();
 
    EXPECT_THROW({
        auto Y = X.view(-1, -1, 2);
    }, Exception);
 
    EXPECT_THROW({
        auto Y = X.view(5, -1);  
    }, Exception);
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

TEST(TestTensorSqueeze, TestBasicSqueeze) {
    Tensor X(float32, {1, 3, 1}); 
    X.initialize();

    X[0][0][0] = 10;
    X[0][1][0] = 20;
    X[0][2][0] = 30;

    Tensor Y = squeeze(X);
 
    ASSERT_EQ(Y.shape().rank(), 1);
    ASSERT_EQ(Y.shape()[0], 3);

    EXPECT_EQ(Y[0], 10);
    EXPECT_EQ(Y[1], 20);
    EXPECT_EQ(Y[2], 30);
 
    Y[1] = 99;
    EXPECT_EQ(X[0][1][0], 99);
}


TEST(TestTensorUnsqueeze, TestUnsqueezeFront) {
    Tensor X(float32, {3}); 
    X.initialize();

    X[0] = 1;
    X[1] = 2;
    X[2] = 3;

    Tensor Y = unsqueeze(X, 0); 

    ASSERT_EQ(Y.shape().rank(), 2);
    ASSERT_EQ(Y.shape()[0], 1);
    ASSERT_EQ(Y.shape()[1], 3);

    EXPECT_EQ(Y[0][0], 1);
    EXPECT_EQ(Y[0][1], 2);
    EXPECT_EQ(Y[0][2], 3);
 
    Y[0][2] = 77;
    EXPECT_EQ(X[2], 77);
}


TEST(TestTensorUnsqueeze, TestUnsqueezeBack) {
    Tensor X(float32, {3}); 
    X.initialize();

    X[0] = 5;
    X[1] = 6;
    X[2] = 7;

    Tensor Y = unsqueeze(X, -1); 

    ASSERT_EQ(Y.shape().rank(), 2);
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 1);

    EXPECT_EQ(Y[0][0], 5);
    EXPECT_EQ(Y[1][0], 6);
    EXPECT_EQ(Y[2][0], 7);
 
    Y[2][0] = 123;
    EXPECT_EQ(X[2], 123);
}


TEST(TestTensorUnsqueeze, TestMultipleAxes) {
    Tensor X(float32, {2, 2}); 
    X.initialize();

    X[0][0] = 1; X[0][1] = 2;
    X[1][0] = 3; X[1][1] = 4;

    Tensor Y = unsqueeze(X, 0, 2);  

    ASSERT_EQ(Y.shape().rank(), 4);
    ASSERT_EQ(Y.shape()[0], 1);
    ASSERT_EQ(Y.shape()[1], 2);
    ASSERT_EQ(Y.shape()[2], 1);
    ASSERT_EQ(Y.shape()[3], 2);

    EXPECT_EQ(Y[0][0][0][0], 1);
    EXPECT_EQ(Y[0][0][0][1], 2);
    EXPECT_EQ(Y[0][1][0][0], 3);
    EXPECT_EQ(Y[0][1][0][1], 4);
 
    Y[0][1][0][1] = 99;
    EXPECT_EQ(X[1][1], 99);
}
