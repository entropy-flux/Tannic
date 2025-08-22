#ifdef CUDA
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "tensor.hpp"
#include "transformations.hpp"

using namespace tannic;  

TEST(TestCUDAOuter, CUDAOuterProduct) {  
    Tensor A(float32, {2}); A.initialize(Device(0)); 
    Tensor B(float32, {3}); B.initialize(Device(0)); 
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

    float host_result[2][3];
    cudaMemcpy(host_result, C.bytes(), sizeof(host_result), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(host_result[i][j], expected[i][j]);
        }
    }
    
}
#endif
