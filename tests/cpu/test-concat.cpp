#include <gtest/gtest.h>
#include "tensor.hpp"
#include "transformations.hpp"
#include "comparisons.hpp"

using namespace tannic;

TEST(TestConcat, ConcatDim0) {
    Tensor A(float32, {2, 2}); 
    A.initialize({{1.0f, 2.0f}, {3.0f, 4.0f}});
    
    Tensor B(float32, {3, 2}); 
    B.initialize({{5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}});

    Tensor Y = concatenate(A, B, 0);

    Tensor expected(float32, {5, 2});
    expected.initialize({
        {1.0f, 2.0f},
        {3.0f, 4.0f},
        {5.0f, 6.0f},
        {7.0f, 8.0f},
        {9.0f, 10.0f}
    });

    EXPECT_EQ(Y.shape(), expected.shape());
    EXPECT_TRUE(allclose(Y, expected));
}

TEST(TestConcat, ConcatDim1) {
    Tensor A(float32, {2, 2}); 
    A.initialize({{1.0f, 2.0f}, {3.0f, 4.0f}});
    
    Tensor B(float32, {2, 3}); 
    B.initialize({{5.0f, 6.0f, 7.0f}, {8.0f, 9.0f, 10.0f}});

    Tensor Y = concatenate(A, B, 1);

    Tensor expected(float32, {2, 5});
    expected.initialize({
        {1.0f, 2.0f, 5.0f, 6.0f, 7.0f},
        {3.0f, 4.0f, 8.0f, 9.0f, 10.0f}
    });

    EXPECT_EQ(Y.shape(), expected.shape());
    EXPECT_TRUE(allclose(Y, expected));
}

TEST(TestConcat, TestNonContiguous) {
    // issued from transformers
    Tensor cls = {{{100.0f, 200.0f, 300.0f}}};
    Tensor features(float32, {2,3,2,2}); features.initialize({
        {  
            { {1.0f, 2.0f}, {3.0f, 4.0f} },   // channel 0
            { {5.0f, 6.0f}, {7.0f, 8.0f} },   // channel 1
            { {9.0f,10.0f}, {11.0f,12.0f} }   // channel 2
        },
        {   
            { {13.0f,14.0f}, {15.0f,16.0f} }, // channel 0
            { {17.0f,18.0f}, {19.0f,20.0f} }, // channel 1
            { {21.0f,22.0f}, {23.0f,24.0f} }  // channel 2
        }
    });
    Tensor flat = flatten(features, 2);
    Tensor sequence = flat.transpose(1,2);
    size_t batch_size = sequence.size(0);
    Tensor expanded = expand(cls, batch_size, -1, -1);
    Tensor out = concatenate(expanded, sequence,1);
  
    Tensor out_expected(float32, {2, 5, 3});
    out_expected.initialize({
        { {100.0f, 200.0f, 300.0f}, {  1.0f,  5.0f,  9.0f}, {  2.0f,  6.0f, 10.0f}, {  3.0f,  7.0f, 11.0f}, {  4.0f,  8.0f, 12.0f} },
        { {100.0f, 200.0f, 300.0f}, { 13.0f, 17.0f, 21.0f}, { 14.0f, 18.0f, 22.0f}, { 15.0f, 19.0f, 23.0f}, { 16.0f, 20.0f, 24.0f} }
    });

    EXPECT_TRUE(allclose(out, out_expected));
}
 