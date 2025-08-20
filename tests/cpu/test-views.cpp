#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Transformations.hpp"

using namespace tannic;

TEST(TestTensorView, TestBasicView) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;
 
    Tensor Y = view(X, 3, 2);
 
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
