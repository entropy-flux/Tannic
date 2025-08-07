#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Transformations.hpp"

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

    float expected[2][3] = {
        {  6.0f, 0.0f, -1.5f },
        { -8.0f, 0.0f,  2.0f }
    };

    float* result = reinterpret_cast<float*>(C.bytes());
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(result[i * 3 + j], expected[i][j]);
        }
    }
}