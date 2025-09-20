#include <gtest/gtest.h>
#include "tensor.hpp"
#include "convolutions.hpp"
#include "comparisons.hpp"

using namespace tannic;

//
// 1D Convolution Tests
//
TEST(TestConvolution1D, Simple1D) { 
    Tensor input(float32, {1, 1, 5});
    input.initialize({{{1, 2, 3, 4, 5}}});

    Tensor kernel(float32, {1, 1, 3});
    kernel.initialize({{{1, 0, -1}}});

    Tensor output = convolve1D(input, kernel, 1, 0);

    Tensor expected(float32, {1, 1, 3});
    expected.initialize({{{-2, -2, -2}}});

    EXPECT_EQ(output.shape(), Shape({1, 1, 3}));
    EXPECT_TRUE(allclose(output, expected));
}

TEST(TestConvolution1D, Stride2) {
    Tensor input(float32, {1, 1, 6});
    input.initialize({{{1, 2, 3, 4, 5, 6}}});

    Tensor kernel(float32, {1, 1, 3});
    kernel.initialize({{{1, 0, -1}}});

    Tensor output = convolve1D(input, kernel, 2, 0);

    Tensor expected(float32, {1, 1, 2});
    expected.initialize({{{-2, -2}}});

    EXPECT_EQ(output.shape(), Shape({1, 1, 2}));
    EXPECT_TRUE(allclose(output, expected));
}

TEST(TestConvolution1D, Padding1) {
    Tensor input(float32, {1, 1, 3});
    input.initialize({{{1, 2, 3}}});

    Tensor kernel(float32, {1, 1, 2});
    kernel.initialize({{{1, -1}}});

    Tensor output = convolve1D(input, kernel, 1, 1);
    Tensor expected(float32, {1, 1, 4}); expected.initialize({{{-1, -1, -1, 3}}});
    EXPECT_EQ(output.shape(), Shape({1, 1, 4}));
    EXPECT_TRUE(allclose(expected, output)); 
}


TEST(TestConvolution1D, MultiChannelInput) {
    Tensor input(float32, {1, 2, 4});  // shape: [batch=1, channels=2, length=4]
    input.initialize({
        {
            {1, 2, 3, 4},   // channel 0
            {5, 6, 7, 8}    // channel 1
        }
    });

    Tensor kernel(float32, {1, 2, 3});  // shape: [out_channels=1, in_channels=2, kernel=3]
    kernel.initialize({
        {   // out_channel 0
            {1, 1, 1},      // in_channel 0
            {1, 1, 1}       // in_channel 1
        }
    });

    Tensor output = convolve1D(input, kernel, 1, 0);
    Tensor expected(float32, {1,1,2}); expected.initialize({{{24, 30}}});
    EXPECT_EQ(output.shape(), Shape({1, 1, 2}));
    EXPECT_TRUE(allclose(output, expected));
}

TEST(TestConvolution1D, Kernel1x1) {
    Tensor input(float32, {1, 1, 4});
    input.initialize({{{1, 2, 3, 4}}});

    Tensor kernel(float32, {1, 1, 1});
    kernel.initialize({{{2}}});

    Tensor output = convolve1D(input, kernel, 1, 0);

    Tensor expected(float32, {1, 1, 4});
    expected.initialize({{{2, 4, 6, 8}}});

    EXPECT_EQ(output.shape(), Shape({1, 1, 4}));
    EXPECT_TRUE(allclose(output, expected));
}
TEST(TestConvolution1D, MultiOutputChannels) {
    Tensor input(float32, {1, 1, 5});  // shape: [batch=1, channels=1, length=5]
    input.initialize({
        {   // batch 0
            {1, 2, 3, 4, 5}   // channel 0
        }
    });

    Tensor kernel(float32, {2, 1, 3});  // shape: [out_channels=2, in_channels=1, kernel=3]
    kernel.initialize({
        {   // out_channel 0
            {1, 0, -1}        // in_channel 0
        },
        {   // out_channel 1
            {0, 1, 0}         // in_channel 0
        }
    });

    Tensor output = convolve1D(input, kernel, 1, 0);

    Tensor expected(float32, {1, 2, 3});
    expected.initialize({
        {   // batch 0
            {-2, -2, -2},   // out_channel 0
            { 2,  3,  4}    // out_channel 1
        }
    });

    EXPECT_EQ(output.shape(), Shape({1, 2, 3}));
    EXPECT_TRUE(allclose(output, expected));
}

//
// 2D Convolution Tests
//
TEST(TestConvolution, Simple2D) { 
    Tensor input(float32, {1, 1, 3, 3});
    input.initialize({{
        {{1, 2, 3},
         {4, 5, 6},
         {7, 8, 9}}
    }});

    Tensor kernel(float32, {1, 1, 2, 2});
    kernel.initialize({{
        {{ 1, 0},
         { 0,-1}}
    }});

    Tensor output = convolve2D(input, kernel, {1,1}, {0,0});

    Tensor expected(float32, {1, 1, 2, 2});
    expected.initialize({{
        {{-4, -4},
         {-4, -4}}
    }});

    EXPECT_EQ(output.shape(), Shape({1, 1, 2, 2}));
    EXPECT_TRUE(allclose(output, expected));
}

TEST(TestConvolution, Stride2) {
    Tensor input(float32, {1,1,4,4});
    input.initialize({{{
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16}
    }}});

    Tensor kernel(float32, {1,1,2,2});
    kernel.initialize({{{
        { 1, 0},
        { 0,-1}
    }}});

    Tensor output = convolve2D(input, kernel, {2,2}, {0,0});
    Tensor expected(float32, {1, 1, 2, 2}); expected.initialize({{{{-5, -5},{-5, -5}}}});
    EXPECT_EQ(output.shape(), Shape({1,1,2,2}));
    EXPECT_TRUE(allclose(expected, output));
}

TEST(TestConvolution, Padding1) {
    Tensor input(float32, {1,1,3,3});
    input.initialize({{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }}});

    Tensor kernel(float32, {1,1,2,2});
    kernel.initialize({{{
        { 1, 0},
        { 0,-1}
    }}});

    Tensor output = convolve2D(input, kernel, {1,1}, {1,1});
    EXPECT_EQ(output.shape(), Shape({1,1,4,4}));
}

TEST(TestConvolution, MultiChannelInput) {
    Tensor input(float32, {1,2,3,3});
    input.initialize({
        {   // batch 0
            {   // channel 0
                {1.f, 2.f, 3.f},
                {4.f, 5.f, 6.f},
                {7.f, 8.f, 9.f}
            },
            {   // channel 1
                {10.f, 11.f, 12.f},
                {13.f, 14.f, 15.f},
                {16.f, 17.f, 18.f}
            }
        }
    });


    Tensor kernel(float32, {1,2,2,2});
    kernel.initialize({
        {   // out_channel 0
            {   // in_channel 0
                {1.f, 1.f},
                {1.f, 1.f}
            },
            {   // in_channel 1
                {1.f, 1.f},
                {1.f, 1.f}
            }
        }
    });


    Tensor output = convolve2D(input, kernel, {1,1}, {0,0});
    EXPECT_EQ(output.shape(), Shape({1,1,2,2}));
}

TEST(TestConvolution, Kernel1x1) {
    Tensor input(float32, {1,1,3,3});
    input.initialize({{{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }}});

    Tensor kernel(float32, {1,1,1,1});
    kernel.initialize({{{{2}}}});

    Tensor output = convolve2D(input, kernel, {1,1}, {0,0});

    Tensor expected(float32, {1,1,3,3});
    expected.initialize({{{
        { 2,  4,  6},
        { 8, 10, 12},
        {14, 16, 18}
    }}});

    EXPECT_TRUE(allclose(output, expected));
}

TEST(TestConvolution, BigConv) {

    Tensor X = {
        { // batch 0
            { // channel 0
                {1.0f,  2.0f,  3.0f,  4.0f},
                {5.0f,  6.0f,  7.0f,  8.0f},
                {9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f,14.0f, 15.0f, 16.0f}
            },
            { // channel 1
                {17.0f, 18.0f, 19.0f, 20.0f},
                {21.0f, 22.0f, 23.0f, 24.0f},
                {25.0f, 26.0f, 27.0f, 28.0f},
                {29.0f, 30.0f, 31.0f, 32.0f}
            },
            { // channel 2
                {33.0f, 34.0f, 35.0f, 36.0f},
                {37.0f, 38.0f, 39.0f, 40.0f},
                {41.0f, 42.0f, 43.0f, 44.0f},
                {45.0f, 46.0f, 47.0f, 48.0f}
            }
        },
        { // batch 1
            { // channel 0
                {49.0f, 50.0f, 51.0f, 52.0f},
                {53.0f, 54.0f, 55.0f, 56.0f},
                {57.0f, 58.0f, 59.0f, 60.0f},
                {61.0f, 62.0f, 63.0f, 64.0f}
            },
            { // channel 1
                {65.0f, 66.0f, 67.0f, 68.0f},
                {69.0f, 70.0f, 71.0f, 72.0f},
                {73.0f, 74.0f, 75.0f, 76.0f},
                {77.0f, 78.0f, 79.0f, 80.0f}
            },
            { // channel 2
                {81.0f, 82.0f, 83.0f, 84.0f},
                {85.0f, 86.0f, 87.0f, 88.0f},
                {89.0f, 90.0f, 91.0f, 92.0f},
                {93.0f, 94.0f, 95.0f, 96.0f}
            }
        }
    };
    
    
    Tensor K = {
        { // out_channel 0
            { // in channel 0
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            },
            { // in channel 1
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            },
            { // in channel 2
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            }
        }
    };

    Tensor Y = convolve2D(X, K, /*stride=*/1, /*padding=*/1) ; 

    Tensor Y_expected(float32, {2, 1, 4, 4});
    Y_expected.initialize({
        {   // first batch
            {
                { -120, -12, -12, 126 },
                { -198, -18, -18, 207 },
                { -234, -18, -18, 243 },
                { -168, -12, -12, 174 }
            }
        },
        {   // second batch
            {
                { -408, -12, -12, 414 },
                { -630, -18, -18, 639 },
                { -666, -18, -18, 675 },
                { -456, -12, -12, 462 }
            }
        }
    });

    EXPECT_TRUE(allclose(Y, Y_expected));
}

TEST(BiasedConvolution, BigBiasedCong) {
    Tensor X(float32, {2,3,4,4}); 
    
    X.initialize({
        { // batch 0
            { // channel 0
                {1.0f,  2.0f,  3.0f,  4.0f},
                {5.0f,  6.0f,  7.0f,  8.0f},
                {9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f,14.0f, 15.0f, 16.0f}
            },
            { // channel 1
                {17.0f, 18.0f, 19.0f, 20.0f},
                {21.0f, 22.0f, 23.0f, 24.0f},
                {25.0f, 26.0f, 27.0f, 28.0f},
                {29.0f, 30.0f, 31.0f, 32.0f}
            },
            { // channel 2
                {33.0f, 34.0f, 35.0f, 36.0f},
                {37.0f, 38.0f, 39.0f, 40.0f},
                {41.0f, 42.0f, 43.0f, 44.0f},
                {45.0f, 46.0f, 47.0f, 48.0f}
            }
        },
        { // batch 1
            { // channel 0
                {49.0f, 50.0f, 51.0f, 52.0f},
                {53.0f, 54.0f, 55.0f, 56.0f},
                {57.0f, 58.0f, 59.0f, 60.0f},
                {61.0f, 62.0f, 63.0f, 64.0f}
            },
            { // channel 1
                {65.0f, 66.0f, 67.0f, 68.0f},
                {69.0f, 70.0f, 71.0f, 72.0f},
                {73.0f, 74.0f, 75.0f, 76.0f},
                {77.0f, 78.0f, 79.0f, 80.0f}
            },
            { // channel 2
                {81.0f, 82.0f, 83.0f, 84.0f},
                {85.0f, 86.0f, 87.0f, 88.0f},
                {89.0f, 90.0f, 91.0f, 92.0f},
                {93.0f, 94.0f, 95.0f, 96.0f}
            }
        }
    });
    
    
    Tensor K(float32, {1,3,3,3}); K.initialize({
        { // out_channel 0
            { // in channel 0
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            },
            { // in channel 1
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            },
            { // in channel 2
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f},
                {1.0f,  0.0f, -1.0f}
            }
        }
    });

    Tensor b(float32, {1});
    b[0] = 10;

    Tensor Y_expected(float32, {2, 1, 4, 4});
    Y_expected.initialize({
        { 
            { 
                {-110.0f,  -2.0f,  -2.0f, 136.0f},
                {-188.0f,  -8.0f,  -8.0f, 217.0f},
                {-224.0f,  -8.0f,  -8.0f, 253.0f},
                {-158.0f,  -2.0f,  -2.0f, 184.0f}
            }
        },
        { 
            { 
                {-398.0f,  -2.0f,  -2.0f, 424.0f},
                {-620.0f,  -8.0f,  -8.0f, 649.0f},
                {-656.0f,  -8.0f,  -8.0f, 685.0f},
                {-446.0f,  -2.0f,  -2.0f, 472.0f}
            }
        }
    });

    Tensor Y = convolve2D(X, K, b, /*stride=*/1, /*padding=*/1) ; 

    EXPECT_TRUE(allclose(Y, Y_expected));

}

TEST(BiasedConvolution, BigBiasedConv2OutChannels) {
    Tensor X(float32, {2,3,4,4});

    X.initialize({
        { // batch 0
            { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} },
            { {17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32} },
            { {33, 34, 35, 36}, {37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48} }
        },
        { // batch 1
            { {49, 50, 51, 52}, {53, 54, 55, 56}, {57, 58, 59, 60}, {61, 62, 63, 64} },
            { {65, 66, 67, 68}, {69, 70, 71, 72}, {73, 74, 75, 76}, {77, 78, 79, 80} },
            { {81, 82, 83, 84}, {85, 86, 87, 88}, {89, 90, 91, 92}, {93, 94, 95, 96} }
        }
    });

    Tensor K(float32, {2,3,3,3});
    K.initialize({
        { // out_channel 0
            { {1,0,-1}, {1,0,-1}, {1,0,-1} },
            { {1,0,-1}, {1,0,-1}, {1,0,-1} },
            { {1,0,-1}, {1,0,-1}, {1,0,-1} }
        },
        { // out_channel 1
            { {-1,0,1}, {-1,0,1}, {-1,0,1} },
            { {-1,0,1}, {-1,0,1}, {-1,0,1} },
            { {-1,0,1}, {-1,0,1}, {-1,0,1} }
        }
    });

    Tensor b(float32, {2});
    b[0] = 10.0f;  // bias for out_channel 0
    b[1] = -5.0f;  // bias for out_channel 1

    Tensor Y_expected(float32, {2,2,4,4});
    Y_expected.initialize({
        { // batch 0
            { {-110, -2, -2, 136}, {-188, -8, -8, 217}, {-224, -8, -8, 253}, {-158, -2, -2, 184} },
            { {115, 7, 7, -131}, {193, 13, 13, -212}, {229, 13, 13, -248}, {163, 7, 7, -179} }
        },
        { // batch 1
            { {-398, -2, -2, 424}, {-620, -8, -8, 649}, {-656, -8, -8, 685}, {-446, -2, -2, 472} },
            { {403, 7, 7, -419}, {625, 13, 13, -644}, {661, 13, 13, -680}, {451, 7, 7, -467} }
        }
    });

    Tensor Y = convolve2D(X, K, b, /*stride=*/1, /*padding=*/1);

    EXPECT_TRUE(allclose(Y, Y_expected));
}
