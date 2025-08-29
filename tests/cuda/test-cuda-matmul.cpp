#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "tensor.hpp"
#include "transformations.hpp"
#include "comparisons.hpp"

using namespace tannic;

TEST(MatmulTestsDVC, BatchedDVC) {
    Tensor A(float32, {1, 2, 2, 2});
    A.initialize({
        {
            { {1, 2}, {3, 4} },
            { {5, 6}, {7, 8} }
        }
    }, Device());

    Tensor B(float32, {1, 2, 2, 2});
    B.initialize({
        {
            { {9, 8}, {7, 6} },
            { {5, 4}, {3, 2} }
        }
    }, Device());

    Tensor Z_expected(float32, {1, 2, 2, 2});
    Z_expected.initialize({
        {
            { {23, 20}, {55, 48} },
            { {43, 32}, {59, 44} }
        }
    }, Device());

    Tensor Z = matmul(A, B);
    EXPECT_TRUE(allclose(Z, Z_expected));
}

TEST(MatmulTestsDVC, SecondTransposedDVC) {
    Tensor X(float32, {2, 3});
    X.initialize({
        {1, 2, 3},
        {4, 5, 6}
    }, Device());

    Tensor Y(float32, {2, 3});
    Y.initialize({
        {7, 8, 9},
        {10, 11, 12}
    }, Device());

    Tensor Z_expected(float32, {2, 2});
    Z_expected.initialize({
        {50, 68},
        {122, 167}
    }, Device());

    Tensor Z = matmul(X, Y.transpose(-1, -2));
    EXPECT_TRUE(allclose(Z, Z_expected));
}

TEST(MatmulTestsDVC, BothTransposedDVC) {
    Tensor X(float32, {3, 2});
    X.initialize({
        {1, 4},
        {2, 5},
        {3, 6}
    }, Device());

    Tensor Y(float32, {2, 3});
    Y.initialize({
        {7, 8, 9},
        {10, 11, 12}
    }, Device());

    Tensor Z_expected(float32, {2, 2});
    Z_expected.initialize({
        {50, 68},
        {122, 167}
    }, Device());

    Tensor Z = matmul(X.transpose(-1, -2), Y.transpose(-1, -2));
    EXPECT_TRUE(allclose(Z, Z_expected));
}

TEST(MatmulTestsDVC, Rank4_SecondTransposedDVC) {
    Tensor X(float32, {2, 2, 2, 4});
    X.initialize({
        {
            { {1,2,3,4}, {5,6,7,8} },
            { {1,2,3,4}, {5,6,7,8} }
        },
        {
            { {1,2,3,4}, {5,6,7,8} },
            { {1,2,3,4}, {5,6,7,8} }
        }
    }, Device());

    Tensor Y(float32, {2, 2, 3, 4});
    Y.initialize({
        {
            {
                {1,2,3,4}, {5,6,7,8}, {9,10,11,12}
            },
            {
                {13,14,15,16}, {17,18,19,20}, {21,22,23,24}
            }
        },
        {
            {
                {1,0,1,0}, {0,1,0,1}, {1,1,1,1}
            },
            {
                {2,2,2,2}, {3,3,3,3}, {4,4,4,4}
            }
        }
    }, Device());

    Tensor Z_expected(float32, {2, 2, 2, 3});
    Z_expected.initialize({
        {
            { {30, 70, 110}, {70, 174, 278} },
            { {150, 190, 230}, {382, 486, 590} }
        },
        {
            { {4, 6, 10}, {12, 14, 26} },
            { {20, 30, 40}, {52, 78, 104} }
        }
    }, Device());

    Tensor Z = matmul(X, Y.transpose(-1, -2));
    EXPECT_TRUE(allclose(Z, Z_expected));
}

TEST(MatmulTestsDVC, Rank4Rank2StridedTransposedDVC) { 
    Tensor X(float32, {2, 3, 4, 4});
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
    }, Device());

    Tensor Y(float32, {6, 4});
    Y.initialize({
        {  1.0f,  2.0f,  3.0f,  4.0f },
        {  5.0f,  6.0f,  7.0f,  8.0f },
        {  9.0f, 10.0f, 11.0f, 12.0f },
        { 13.0f, 14.0f, 15.0f, 16.0f },
        { 17.0f, 18.0f, 19.0f, 20.0f },
        { 21.0f, 22.0f, 23.0f, 24.0f }
    }, Device());

    Tensor Z_expected(float32, {2, 3, 4, 6});
    Z_expected.initialize({
        { 
            { 
                {  30.0f,   70.0f,  110.0f,  150.0f,  190.0f,  230.0f },
                {  70.0f,  174.0f,  278.0f,  382.0f,  486.0f,  590.0f },
                { 110.0f,  278.0f,  446.0f,  614.0f,  782.0f,  950.0f },
                { 150.0f,  382.0f,  614.0f,  846.0f, 1078.0f, 1310.0f }
            },
            { 
                { 190.0f,  486.0f,  782.0f, 1078.0f, 1374.0f, 1670.0f },
                { 230.0f,  590.0f,  950.0f, 1310.0f, 1670.0f, 2030.0f },
                { 270.0f,  694.0f, 1118.0f, 1542.0f, 1966.0f, 2390.0f },
                { 310.0f,  798.0f, 1286.0f, 1774.0f, 2262.0f, 2750.0f }
            },
            {
                { 350.0f,  902.0f, 1454.0f, 2006.0f, 2558.0f, 3110.0f },
                { 390.0f, 1006.0f, 1622.0f, 2238.0f, 2854.0f, 3470.0f },
                { 430.0f, 1110.0f, 1790.0f, 2470.0f, 3150.0f, 3830.0f },
                { 470.0f, 1214.0f, 1958.0f, 2702.0f, 3446.0f, 4190.0f }
            }
        },
        {
            { 
                { 510.0f, 1318.0f, 2126.0f, 2934.0f, 3742.0f, 4550.0f },
                { 550.0f, 1422.0f, 2294.0f, 3166.0f, 4038.0f, 4910.0f },
                { 590.0f, 1526.0f, 2462.0f, 3398.0f, 4334.0f, 5270.0f },
                { 630.0f, 1630.0f, 2630.0f, 3630.0f, 4630.0f, 5630.0f }
            },
            { 
                { 670.0f, 1734.0f, 2798.0f, 3862.0f, 4926.0f, 5990.0f },
                { 710.0f, 1838.0f, 2966.0f, 4094.0f, 5222.0f, 6350.0f },
                { 750.0f, 1942.0f, 3134.0f, 4326.0f, 5518.0f, 6710.0f },
                { 790.0f, 2046.0f, 3302.0f, 4558.0f, 5814.0f, 7070.0f }
            },
            { 
                { 830.0f, 2150.0f, 3470.0f, 4790.0f, 6110.0f, 7430.0f },
                { 870.0f, 2254.0f, 3638.0f, 5022.0f, 6406.0f, 7790.0f },
                { 910.0f, 2358.0f, 3806.0f, 5254.0f, 6702.0f, 8150.0f },
                { 950.0f, 2462.0f, 3974.0f, 5486.0f, 6998.0f, 8510.0f }
            }
        }
    }, Device());

    Tensor Z = matmul(X, Y.transpose());
    EXPECT_TRUE(allclose(Z, Z_expected));
}

#endif