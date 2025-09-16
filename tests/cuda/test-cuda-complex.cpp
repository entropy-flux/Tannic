#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath> 

#include "tensor.hpp"
#include "complex.hpp" 
#include "comparisons.hpp"

using namespace tannic;

class CUDAComplexTests : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(CUDAComplexTests, CUDAComplexificationAndRealification) {
    Tensor X(float32, {2,2});  X.initialize(Device());   
    X[0][0] = 1;
    X[0][1] = 0;
    X[1][0] = 2;
    X[1][1] = 3;   
    Tensor Z = complexify(X);   

    Tensor Y = realify(Z);  
    ASSERT_EQ(X[0][0], 1);
    ASSERT_EQ(X[0][1], 0); 
    ASSERT_EQ(X[1][0], 2); 
    ASSERT_EQ(X[1][1], 3); 
}   

TEST_F(CUDAComplexTests, CUDAPolarConversionManual) { 
    Tensor X(float32, {2, 2});
    X.initialize(Device()); 

    X[0][0] = 1.0f;  X[0][1] = 6.0f;
    X[1][0] = 2.0f;  X[1][1] = 3.0f;
 
    Tensor Y(float32, {2, 2});
    Y.initialize(Device());

    Y[0][0] = 2.0f;  Y[0][1] = 1.0f;
    Y[1][0] = 1.5f;  Y[1][1] = 3.14f;
 
    Tensor Z = polar(X, Y);  
    Tensor W = realify(Z);  

    Tensor expected(float32, {2, 2, 2}); expected.initialize({
        {{-0.416147, 0.909297}, {3.24181, 5.04883}},
        {{0.141474, 1.99499}, {-3.0, 0.00477764}}
    }, Device());
 
    EXPECT_TRUE(allclose(W, expected));
}

#endif


/*

import torch

# Create magnitude tensor X
X = torch.tensor([[1.0, 6.0],
                  [2.0, 3.0]])

# Create phase tensor Y (in radians)
Y = torch.tensor([[2.0, 1.0],
                  [1.5, 3.14]])

# Create complex tensor Z from polar coordinates
Z = torch.polar(X, Y)

print("Z real part:\n", Z.real)
print("Z imag part:\n", Z.imag)
 
Z real part:
 tensor([[-0.4161,  3.2418],
        [ 0.1415, -3.0000]])
Z imag part:
 tensor([[9.0930e-01, 5.0488e+00],
        [1.9950e+00, 4.7776e-03]])


*/