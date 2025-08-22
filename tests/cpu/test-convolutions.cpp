#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Convolutions.hpp"

using namespace tannic;

#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Convolutions.hpp"

using namespace tannic;

TEST(TestConvolution1D, Simple1D) { 
    Tensor input(float32, {1, 1, 5});   
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 5; ++i) {
        in_data[i] = static_cast<float>(i + 1);   
    }
 
    Tensor kernel(float32, {1, 1, 3});   
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1.0f; k_data[1] = 0.0f; k_data[2] = -1.0f;

    Tensor output = convolve<1>(input, kernel, 1, 0);
    float* out_data = reinterpret_cast<float*>(output.bytes());
 
    float expected[] = {-2.0f, -2.0f, -2.0f};

    ASSERT_EQ(output.shape(), Shape({1, 1, 3}));
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], expected[i]) 
            << "Mismatch at output[" << i << "]";
    }
}

TEST(TestConvolution1D, Stride2) {
    Tensor input(float32, {1, 1, 6});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 6; ++i) in_data[i] = static_cast<float>(i + 1); 

    Tensor kernel(float32, {1, 1, 3});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1; k_data[1] = 0; k_data[2] = -1;

    Tensor output = convolve<1>(input, kernel, 2, 0);
    ASSERT_EQ(output.shape(), Shape({1, 1, 2}));   
    
    float* out_data = reinterpret_cast<float*>(output.bytes()); 
    EXPECT_FLOAT_EQ(out_data[0], -2.0f);
    EXPECT_FLOAT_EQ(out_data[1], -2.0f);
}
 
TEST(TestConvolution1D, Padding1) {
    Tensor input(float32, {1, 1, 3});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 3; ++i) in_data[i] = static_cast<float>(i + 1);  

    Tensor kernel(float32, {1, 1, 2});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1; k_data[1] = -1;

    Tensor output = convolve<1>(input, kernel, 1, 1);
    ASSERT_EQ(output.shape(), Shape({1, 1, 4}));   
}
 
TEST(TestConvolution1D, MultiChannelInput) {
    Tensor input(float32, {1, 2, 4});  // 2 input channels
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 8; ++i) in_data[i] = static_cast<float>(i + 1);

    Tensor kernel(float32, {1, 2, 3});  
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    for (int i = 0; i < 6; ++i) k_data[i] = 1.0f;

    Tensor output = convolve<1>(input, kernel, 1, 0);
    ASSERT_EQ(output.shape(), Shape({1, 1, 2}));   
}
 
TEST(TestConvolution1D, Kernel1x1) {
    Tensor input(float32, {1, 1, 4});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 4; ++i) in_data[i] = static_cast<float>(i + 1); 

    Tensor kernel(float32, {1, 1, 1});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 2.0f;

    Tensor output = convolve<1>(input, kernel, 1, 0);
    float* out_data = reinterpret_cast<float*>(output.bytes());
    ASSERT_EQ(output.shape(), Shape({1, 1, 4}));
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], in_data[i] * 2.0f);
    }
}

TEST(TestConvolution1D, MultiOutputChannels) {
    Tensor input(float32, {1, 1, 5});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 5; ++i) in_data[i] = static_cast<float>(i + 1);  

    Tensor kernel(float32, {2, 1, 3});  
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes()); 
    k_data[0] = 1.0f; k_data[1] = 0.0f; k_data[2] = -1.0f; 
    k_data[3] = 0.0f; k_data[4] = 1.0f; k_data[5] = 0.0f;

    Tensor output = convolve<1>(input, kernel, 1, 0);
    ASSERT_EQ(output.shape(), Shape({1, 2, 3}));  
    
    float* out_data = reinterpret_cast<float*>(output.bytes()); 
    EXPECT_FLOAT_EQ(out_data[0], -2.0f);
    EXPECT_FLOAT_EQ(out_data[1], -2.0f);
    EXPECT_FLOAT_EQ(out_data[2], -2.0f);
     
    EXPECT_FLOAT_EQ(out_data[3], 2.0f);
    EXPECT_FLOAT_EQ(out_data[4], 3.0f);
    EXPECT_FLOAT_EQ(out_data[5], 4.0f);
}


TEST(TestConvolution, Simple2D) { 
    Tensor input(float32, {1, 1, 3, 3});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 9; ++i) {
        in_data[i] = static_cast<float>(i + 1);  // 1,2,...,9
    }
 
    Tensor kernel(float32, {1, 1, 2, 2});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1.0f; k_data[1] = 0.0f;
    k_data[2] = 0.0f; k_data[3] = -1.0f; 

    Tensor output = convolve<2>(input, kernel, {1,1}, {0,0});
    float* out_data = reinterpret_cast<float*>(output.bytes());

    // Expected 2x2 result
    float expected[] = {
        -4.0f, -4.0f,
        -4.0f, -4.0f
    };

    ASSERT_EQ(output.shape(), Shape({1, 1, 2, 2}));
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], expected[i]) 
            << "Mismatch at output[" << i << "]";
    }
}


TEST(TestConvolution, Stride2) {
    Tensor input(float32, {1,1,4,4});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 16; ++i) in_data[i] = static_cast<float>(i + 1);

    Tensor kernel(float32, {1,1,2,2});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1; k_data[1] = 0;
    k_data[2] = 0; k_data[3] = -1;

    Tensor output = convolve<2>(input, kernel, {2,2}, {0,0});
    ASSERT_EQ(output.shape(), Shape({1,1,2,2}));
}
 
TEST(TestConvolution, Padding1) {
    Tensor input(float32, {1,1,3,3});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 9; ++i) in_data[i] = static_cast<float>(i + 1);

    Tensor kernel(float32, {1,1,2,2});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 1; k_data[1] = 0;
    k_data[2] = 0; k_data[3] = -1;

    Tensor output = convolve<2>(input, kernel, {1,1}, {1,1});
    ASSERT_EQ(output.shape(), Shape({1,1,4,4}));
}
 
TEST(TestConvolution, MultiChannelInput) {
    Tensor input(float32, {1,2,3,3});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 18; ++i) in_data[i] = static_cast<float>(i + 1);

    Tensor kernel(float32, {1,2,2,2});  // 2 input channels
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    for (int i = 0; i < 8; ++i) k_data[i] = 1.0f;

    Tensor output = convolve<2>(input, kernel, {1,1}, {0,0});
    ASSERT_EQ(output.shape(), Shape({1,1,2,2}));
}
 
TEST(TestConvolution, Kernel1x1) {
    Tensor input(float32, {1,1,3,3});
    input.initialize();
    float* in_data = reinterpret_cast<float*>(input.bytes());
    for (int i = 0; i < 9; ++i) in_data[i] = static_cast<float>(i + 1);

    Tensor kernel(float32, {1,1,1,1});
    kernel.initialize();
    float* k_data = reinterpret_cast<float*>(kernel.bytes());
    k_data[0] = 2.0f;

    Tensor output = convolve<2>(input, kernel, {1,1}, {0,0});
    float* out_data = reinterpret_cast<float*>(output.bytes());
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], in_data[i] * 2.0f);
    }
}