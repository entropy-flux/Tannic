#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "Tensor.hpp"
#include "Complex.hpp"
#include "Transformations.hpp"

using namespace tannic;

class ComplexTests : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ComplexTests, ComplexificationAndRealification) {
    Tensor X(float32, {2,2}); X.initialize();  
    X[0][0] = 1;
    X[0][1] = 0;
    X[1][0] = 2;
    X[1][1] = 3;  
    Tensor Z = complex(X);

    float* data = reinterpret_cast<float*>(Z.bytes());
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 0);
    ASSERT_EQ(data[2], 2);
    ASSERT_EQ(data[3], 3);
    
    Tensor Y = real(Z);


    ASSERT_EQ(X[0][0], 1);
    ASSERT_EQ(X[0][1], 0); 
    ASSERT_EQ(X[1][0], 2); 
    ASSERT_EQ(X[1][1], 3); 
} 