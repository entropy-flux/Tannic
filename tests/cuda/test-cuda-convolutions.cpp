#ifdef CUDA
#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Convolutions.hpp"
#include <cuda_runtime.h>
#include <cstring>

using namespace tannic;

TEST(TestCudaConvolution1D, Simple1D) { 
    Tensor input(float32, {1, 1, 5});   
    input.initialize(Device());
     
    std::vector<float> host_input(5);
    for (int i = 0; i < 5; ++i) {
        host_input[i] = static_cast<float>(i + 1);   
    } 
    cudaMemcpy(input.bytes(), host_input.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);
 
    Tensor kernel(float32, {1, 1, 3});   
    kernel.initialize(Device());
     
    std::vector<float> host_kernel = {1.0f, 0.0f, -1.0f}; 
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 0);
     
    std::vector<float> host_output(3);
    cudaMemcpy(host_output.data(), output.bytes(), 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float expected[] = {-2.0f, -2.0f, -2.0f};

    ASSERT_EQ(output.shape(), Shape({1, 1, 3}));
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], expected[i]) 
            << "Mismatch at output[" << i << "]";
    }
}

TEST(TestCudaConvolution1D, Stride2) {
    Tensor input(float32, {1, 1, 6});
    input.initialize(Device());
    
    std::vector<float> host_input(6);
    for (int i = 0; i < 6; ++i) host_input[i] = static_cast<float>(i + 1);
    cudaMemcpy(input.bytes(), host_input.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {1, 1, 3});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel = {1.0f, 0.0f, -1.0f};
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 2, 0);
    ASSERT_EQ(output.shape(), Shape({1, 1, 2}));   
    
    std::vector<float> host_output(2);
    cudaMemcpy(host_output.data(), output.bytes(), 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    EXPECT_FLOAT_EQ(host_output[0], -2.0f);
    EXPECT_FLOAT_EQ(host_output[1], -2.0f);
}
 
TEST(TestCudaConvolution1D, Padding1) {
    Tensor input(float32, {1, 1, 3});
    input.initialize(Device());
    
    std::vector<float> host_input(3);
    for (int i = 0; i < 3; ++i) host_input[i] = static_cast<float>(i + 1);
    cudaMemcpy(input.bytes(), host_input.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {1, 1, 2});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel = {1.0f, -1.0f};
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 2 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 1);
    ASSERT_EQ(output.shape(), Shape({1, 1, 4}));   
    
    // Verify we can copy the result back
    std::vector<float> host_output(4);
    cudaMemcpy(host_output.data(), output.bytes(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
}
 
TEST(TestCudaConvolution1D, MultiChannelInput) {
    Tensor input(float32, {1, 2, 4});
    input.initialize(Device());
    
    std::vector<float> host_input(8);
    for (int i = 0; i < 8; ++i) host_input[i] = static_cast<float>(i + 1);
    cudaMemcpy(input.bytes(), host_input.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {1, 2, 3});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel(6, 1.0f);
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 0);
    ASSERT_EQ(output.shape(), Shape({1, 1, 2}));   
    
    std::vector<float> host_output(2);
    cudaMemcpy(host_output.data(), output.bytes(), 2 * sizeof(float), cudaMemcpyDeviceToHost);
}
 
TEST(TestCudaConvolution1D, Kernel1x1) {
    Tensor input(float32, {1, 1, 4});
    input.initialize(Device());
    
    std::vector<float> host_input(4);
    for (int i = 0; i < 4; ++i) host_input[i] = static_cast<float>(i + 1);
    cudaMemcpy(input.bytes(), host_input.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {1, 1, 1});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel = {2.0f};
    cudaMemcpy(kernel.bytes(), host_kernel.data(), sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 0);
    
    std::vector<float> host_output(4);
    cudaMemcpy(host_output.data(), output.bytes(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    ASSERT_EQ(output.shape(), Shape({1, 1, 4}));
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(host_output[i], host_input[i] * 2.0f);
    }
}

TEST(TestCudaConvolution1D, MultiOutputChannels) {
    Tensor input(float32, {1, 1, 5});
    input.initialize(Device());
    
    std::vector<float> host_input(5);
    for (int i = 0; i < 5; ++i) host_input[i] = static_cast<float>(i + 1);
    cudaMemcpy(input.bytes(), host_input.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {2, 1, 3});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel = {
        1.0f, 0.0f, -1.0f,  // First output channel
        0.0f, 1.0f, 0.0f    // Second output channel
    };
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 0);
    ASSERT_EQ(output.shape(), Shape({1, 2, 3}));  
    
    std::vector<float> host_output(6);
    cudaMemcpy(host_output.data(), output.bytes(), 6 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // First channel
    EXPECT_FLOAT_EQ(host_output[0], -2.0f);
    EXPECT_FLOAT_EQ(host_output[1], -2.0f);
    EXPECT_FLOAT_EQ(host_output[2], -2.0f);
    
    // Second channel
    EXPECT_FLOAT_EQ(host_output[3], 2.0f);
    EXPECT_FLOAT_EQ(host_output[4], 3.0f);
    EXPECT_FLOAT_EQ(host_output[5], 4.0f);
}

TEST(TestCudaConvolution1D, LargeInput) {
    const size_t length = 1024;
    Tensor input(float32, {1, 1, length});
    input.initialize(Device());
    
    std::vector<float> host_input(length);
    for (size_t i = 0; i < length; ++i) {
        host_input[i] = static_cast<float>(i % 10);
    }
    cudaMemcpy(input.bytes(), host_input.data(), length * sizeof(float), cudaMemcpyHostToDevice);

    Tensor kernel(float32, {1, 1, 5});
    kernel.initialize(Device());
    
    std::vector<float> host_kernel = {1.0f, -1.0f, 0.5f, -0.5f, 0.25f};
    cudaMemcpy(kernel.bytes(), host_kernel.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor output = convolve<1>(input, kernel, 1, 2);
    
    size_t expected_length = (length + 2*2 - 5) / 1 + 1;
    ASSERT_EQ(output.shape(), Shape({1, 1, expected_length}));
    
    std::vector<float> host_output(expected_length);
    cudaMemcpy(host_output.data(), output.bytes(), expected_length * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Just verify we got some non-zero results
    bool has_non_zero = false;
    for (size_t i = 0; i < expected_length; ++i) {
        if (host_output[i] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}
#endif