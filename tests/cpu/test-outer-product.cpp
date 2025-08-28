#include <gtest/gtest.h>
#include "tensor.hpp"
#include "comparisons.hpp"
#include "transformations.hpp"

using namespace tannic; 

TEST(TestOuter, HostOuterProduct) {  
    Tensor A(float32, {2}); A.initialize(); // CPU
    Tensor B(float32, {3}); B.initialize(); // CPU
    A[0] = 1.5f;
    A[1] = -2.0f;
    
    B[0] = 4.0f;
    B[1] = 0.0f;
    B[2] = -1.0f;
 
    Tensor C = outer(A, B);   
    Tensor C_expected = { 
        {  6.0f, 0.0f, -1.5f },
        { -8.0f, 0.0f,  2.0f }
    };
    EXPECT_TRUE(allclose(C, C_expected));
}