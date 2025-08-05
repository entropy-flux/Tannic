#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "Tensor.hpp"
#include "Transformations.hpp"

using namespace tannic;

class CUDAMatmulTests : public ::testing::Test {
protected:
    void SetUp() override {}
    
    void compareMatrices(const Tensor& result, const float* expected, int size, float epsilon = 1e-5f) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float* cpu_data = new float[size];
        
        cudaMemcpy(cpu_data, gpu_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < size; ++i) {
            EXPECT_NEAR(cpu_data[i], expected[i], epsilon) << "Mismatch at index " << i;
        }
        
        delete[] cpu_data;
    }
};

TEST_F(CUDAMatmulTests, Basic) {
    Tensor X(float32, {3, 4}); X.initialize(Device(0));
    Tensor Y(float32, {4, 5}); Y.initialize(Device(0));
    
    float X_data[3 * 4] = {
        4.f, 2.f, 6.f, 0.f,
        1.f, 3.f, 5.f, 7.f,
        2.f, 4.f, 6.f, 8.f
    };
    
    float Y_data[4 * 5] = {
        1.f, 2.f, 3.f, 4.f, 5.f,
        0.f, 1.f, 0.f, 1.f, 0.f,
        2.f, 3.f, 4.f, 5.f, 6.f,
        1.f, 0.f, 1.f, 0.f, 1.f
    };
  
    float Z_expected[3 * 5] = {
        16.0f, 28.0f, 36.0f, 48.0f, 56.0f,
        18.0f, 20.0f, 30.0f, 32.0f, 42.0f,
        22.0f, 26.0f, 38.0f, 42.0f, 54.0f
    };
 
    cudaMemcpy(X.bytes(), X_data, 3*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.bytes(), Y_data, 4*5*sizeof(float), cudaMemcpyHostToDevice);

    Tensor Z = matmul(X, Y);   
    compareMatrices(Z, Z_expected, 3*5);
}

TEST_F(CUDAMatmulTests, FirstTransposed) {
    Tensor X(float32, {2, 3}); X.initialize(Device(0));
    Tensor Y(float32, {2, 3}); Y.initialize(Device(0));
     
    float X_data[2 * 3] = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    }; 

    float Y_data[2 * 3] = {
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };
 
    cudaMemcpy(X.bytes(), X_data, 2*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.bytes(), Y_data, 2*3*sizeof(float), cudaMemcpyHostToDevice);
             
    Tensor Z = matmul(X.transpose(-1, -2), Y);

    float Z_expected[3*3] = {
        47.f, 52.f, 57.f,
        64.f, 71.f, 78.f,
        81.f, 90.f, 99.f
    };

    compareMatrices(Z, Z_expected, 3*3);
}

TEST_F(CUDAMatmulTests, Batched) {
    Tensor A(float32, {2, 2, 2}); A.initialize(Device(0));
    Tensor B(float32, {2, 2, 2}); B.initialize(Device(0));
    
    float A_data[2][2][2] = {
        {{1.f, 2.f}, {3.f, 4.f}},
        {{5.f, 6.f}, {7.f, 8.f}}
    };

    float B_data[2][2][2] = {
        {{9.f, 8.f}, {7.f, 6.f}},
        {{5.f, 4.f}, {3.f, 2.f}}
    };

    cudaMemcpy(A.bytes(), A_data, 2*2*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.bytes(), B_data, 2*2*2*sizeof(float), cudaMemcpyHostToDevice);

    Tensor Z = matmul(A, B);
 
    float Z_expected[2][2][2] = {
        {
            {23.f, 20.f}, 
            {55.f, 48.f}
        },
        {
            {43.f, 32.f}, 
            {59.f, 44.f}
        }
    };

    compareMatrices(Z, &Z_expected[0][0][0], 2*2*2);
}

TEST_F(CUDAMatmulTests, SecondTransposed) {
    Tensor X(float32, {2, 3}); X.initialize(Device(0));
    Tensor Y(float32, {2, 3}); Y.initialize(Device(0));
    
    float X_data[2][3] = {
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    };

    float Y_data[2][3] = {
        {7.f, 8.f, 9.f},
        {10.f, 11.f, 12.f}
    };

    cudaMemcpy(X.bytes(), X_data, 2*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.bytes(), Y_data, 2*3*sizeof(float), cudaMemcpyHostToDevice);

    Tensor Z = matmul(X, Y.transpose(-1, -2));

    float Z_expected[2][2] = {
        {50.f, 68.f},
        {122.f, 167.f}
    };

    compareMatrices(Z, &Z_expected[0][0], 2*2);
}

TEST_F(CUDAMatmulTests, BothTransposed) {
    Tensor X(float32, {3, 2}); X.initialize(Device(0));
    Tensor Y(float32, {2, 3}); Y.initialize(Device(0));
    
    float X_data[3][2] = {
        {1.f, 4.f},
        {2.f, 5.f},
        {3.f, 6.f}
    };

    float Y_data[2][3] = {
        {7.f, 8.f, 9.f},
        {10.f, 11.f, 12.f}
    };

    cudaMemcpy(X.bytes(), X_data, 3*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.bytes(), Y_data, 2*3*sizeof(float), cudaMemcpyHostToDevice);

    Tensor Z = matmul(X.transpose(-1, -2), Y.transpose(-1, -2));

    float Z_expected[2][2] = {
        {50.f, 68.f},
        {122.f, 167.f}
    };

    compareMatrices(Z, &Z_expected[0][0], 2*2);
}

TEST_F(CUDAMatmulTests, Rank4_SecondTransposed) {
    Tensor X(float32, {2, 2, 2, 4}); X.initialize(Device(0));
    Tensor Y(float32, {2, 2, 3, 4}); Y.initialize(Device(0));
    
    float X_data[2][2][2][4] = {
        {
            {{1, 2, 3, 4}, {5, 6, 7, 8}},
            {{1, 2, 3, 4}, {5, 6, 7, 8}}
        },
        {
            {{1, 2, 3, 4}, {5, 6, 7, 8}},
            {{1, 2, 3, 4}, {5, 6, 7, 8}}
        }
    };

    float Y_data[2][2][3][4] = {
        {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9,10,11,12}
            },
            {
                {13,14,15,16},
                {17,18,19,20},
                {21,22,23,24}
            }
        },
        {
            {
                {1, 0, 1, 0},
                {0, 1, 0, 1},
                {1, 1, 1, 1}
            },
            {
                {2, 2, 2, 2},
                {3, 3, 3, 3},
                {4, 4, 4, 4}
            }
        }
    };

    cudaMemcpy(X.bytes(), X_data, 2*2*2*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.bytes(), Y_data, 2*2*3*4*sizeof(float), cudaMemcpyHostToDevice);

    Tensor Z = matmul(X, Y.transpose(-1, -2));

    float Z_expected[2][2][2][3] = {
        {
            {{30.f, 70.f, 110.f}, {70.f, 174.f, 278.f}},
            {{150.f, 190.f, 230.f}, {382.f, 486.f, 590.f}}
        },
        {
            {{4.f, 6.f, 10.f}, {12.f, 14.f, 26.f}},
            {{20.f, 30.f, 40.f}, {52.f, 78.f, 104.f}}
        }
    };

    compareMatrices(Z, &Z_expected[0][0][0][0], 2*2*2*3);
}