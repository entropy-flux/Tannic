#include <gtest/gtest.h>
#include "tensor.hpp"
#include "views.hpp"
#include "limits.hpp"
#include "transformations.hpp"
#include "comparisons.hpp"

using namespace tannic;

TEST(TestTensorView, TestBasicView) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;
 
    Tensor Y = X.view(3, 2);
 
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 2);
 
    int expected = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_EQ(Y[i][j], expected)
                << "Mismatch at Y[" << i << "][" << j << "]";
            expected++;
        }
    }
 
    Y[0][0] = 100;
    EXPECT_EQ(X[0][0], 100);
}

TEST(TestTensorReshape, TestBasicReshape) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;
 
    Tensor Y = reshape(X, 3, 2);
 
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 2);
 
    int expected = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_EQ(Y[i][j], expected)
                << "Mismatch at Y[" << i << "][" << j << "]";
            expected++;
        }
    }
 
    Y[0][0] = 100;
    EXPECT_EQ(X[0][0], 1);
}

TEST(TestTensorView, TestViewInferMiddleDimension) {
    Tensor X(float32, {2, 3, 4}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;
 
    Tensor Y = X.view(2, -1, 2);

    ASSERT_EQ(Y.shape().rank(), 3);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 6);
    EXPECT_EQ(Y.shape()[2], 2);
 
    EXPECT_EQ(Y.shape()[0] * Y.shape()[1] * Y.shape()[2], 24);
 
    int expected = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 6; j++)
            for (size_t k = 0; k < 2; k++)
                EXPECT_EQ(Y[i][j][k], expected++);
}

TEST(TestTensorView, TestViewInferFirstDimension) {
    Tensor X(float32, {2, 3, 4}); 
 
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;
 
    Tensor Y = X.view(-1, 3, 4);

    ASSERT_EQ(Y.shape().rank(), 3);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 3);
    EXPECT_EQ(Y.shape()[2], 4);

    int expected = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                EXPECT_EQ(Y[i][j][k], expected++);
}

TEST(TestTensorView, TestViewInferLastDimension) {
    Tensor X(float32, {2, 3, 4});  
    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;

    Tensor Y = X.view(4, -1);

    ASSERT_EQ(Y.shape().rank(), 2);
    EXPECT_EQ(Y.shape()[0], 4);
    EXPECT_EQ(Y.shape()[1], 6);

    int expected = 1;
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(Y[i][j], expected++);
}

TEST(TestTensorView, TestViewInferInvalidMultiple) {
    Tensor X(float32, {2, 3, 4}); 
    X.initialize();
 
    EXPECT_THROW({
        auto Y = X.view(-1, -1, 2);
    }, Exception);
 
    EXPECT_THROW({
        auto Y = X.view(5, -1);  
    }, Exception);
}

TEST(TestTensorExpand, TestBasicExpand) {
    Tensor X(float32, {1, 3}); 
    X.initialize();
 
    X[0][0] = 1;
    X[0][1] = 2;
    X[0][2] = 3;
 
    Tensor Y = expand(X, 4, 3);

    ASSERT_EQ(Y.shape()[0], 4);
    ASSERT_EQ(Y.shape()[1], 3);
 
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(Y[i][0], 1) << "Mismatch at Y[" << i << "][0]";
        EXPECT_EQ(Y[i][1], 2) << "Mismatch at Y[" << i << "][1]";
        EXPECT_EQ(Y[i][2], 3) << "Mismatch at Y[" << i << "][2]";
    }
  
    Y[2][1] = 42;
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(Y[i][1], 42) << "Mismatch at Y[" << i << "][1]";
    }
}

TEST(TestTensorExpand, TestExpandWithNegativeIndex) {
    Tensor X(float32, {2, 1}); 
    X.initialize();

    X[0][0] = 10;
    X[1][0] = 20;
 
    Tensor Y = expand(X, -1, 3);

    ASSERT_EQ(Y.shape().rank(), 2);
    EXPECT_EQ(Y.shape()[0], 2); // kept from X
    EXPECT_EQ(Y.shape()[1], 3); // expanded
 
    EXPECT_EQ(Y[0][0], 10);
    EXPECT_EQ(Y[0][1], 10);
    EXPECT_EQ(Y[0][2], 10);
    EXPECT_EQ(Y[1][0], 20);
    EXPECT_EQ(Y[1][1], 20);
    EXPECT_EQ(Y[1][2], 20);
 
    Y[1][2] = 77;
    EXPECT_EQ(X[1][0], 77);
}

TEST(TestTensorExpand, TestExpandNegativeIndexInvalid) {
    Tensor X(float32, {2, 3}); 
    X.initialize();
 
    EXPECT_THROW({
        auto Y = expand(X, -1, 3, 3);
    }, Exception);
 
    EXPECT_THROW({
        auto Y = expand(X, 5, -1);
    }, Exception);
}

TEST(TestTensorSqueeze, TestBasicSqueeze) {
    Tensor X(float32, {1, 3, 1}); 
    X.initialize();

    X[0][0][0] = 10;
    X[0][1][0] = 20;
    X[0][2][0] = 30;

    Tensor Y = squeeze(X);
 
    ASSERT_EQ(Y.shape().rank(), 1);
    ASSERT_EQ(Y.shape()[0], 3);

    EXPECT_EQ(Y[0], 10);
    EXPECT_EQ(Y[1], 20);
    EXPECT_EQ(Y[2], 30);
 
    Y[1] = 99;
    EXPECT_EQ(X[0][1][0], 99);
}


TEST(TestTensorUnsqueeze, TestUnsqueezeFront) {
    Tensor X(float32, {3}); 
    X.initialize();

    X[0] = 1;
    X[1] = 2;
    X[2] = 3;

    Tensor Y = unsqueeze(X, 0); 

    ASSERT_EQ(Y.shape().rank(), 2);
    ASSERT_EQ(Y.shape()[0], 1);
    ASSERT_EQ(Y.shape()[1], 3);

    EXPECT_EQ(Y[0][0], 1);
    EXPECT_EQ(Y[0][1], 2);
    EXPECT_EQ(Y[0][2], 3);
 
    Y[0][2] = 77;
    EXPECT_EQ(X[2], 77);
}


TEST(TestTensorUnsqueeze, TestUnsqueezeBack) {
    Tensor X(float32, {3}); 
    X.initialize();

    X[0] = 5;
    X[1] = 6;
    X[2] = 7;

    Tensor Y = unsqueeze(X, -1); 

    ASSERT_EQ(Y.shape().rank(), 2);
    ASSERT_EQ(Y.shape()[0], 3);
    ASSERT_EQ(Y.shape()[1], 1);

    EXPECT_EQ(Y[0][0], 5);
    EXPECT_EQ(Y[1][0], 6);
    EXPECT_EQ(Y[2][0], 7);
 
    Y[2][0] = 123;
    EXPECT_EQ(X[2], 123);
}


TEST(TestTensorUnsqueeze, TestMultipleAxes) {
    Tensor X(float32, {2, 2}); 
    X.initialize();

    X[0][0] = 1; X[0][1] = 2;
    X[1][0] = 3; X[1][1] = 4;

    Tensor Y = unsqueeze(X, 0, 2);  

    ASSERT_EQ(Y.shape().rank(), 4);
    ASSERT_EQ(Y.shape()[0], 1);
    ASSERT_EQ(Y.shape()[1], 2);
    ASSERT_EQ(Y.shape()[2], 1);
    ASSERT_EQ(Y.shape()[3], 2);

    EXPECT_EQ(Y[0][0][0][0], 1);
    EXPECT_EQ(Y[0][0][0][1], 2);
    EXPECT_EQ(Y[0][1][0][0], 3);
    EXPECT_EQ(Y[0][1][0][1], 4);
 
    Y[0][1][0][1] = 99;
    EXPECT_EQ(X[1][1], 99);
}

TEST(TestTensorFlatten, TestFlattenAllDims) {
    Tensor X(float32, {2, 3}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            X[i][j] = val++;

    Tensor Y = flatten(X);  

    ASSERT_EQ(Y.shape().rank(), 1);
    ASSERT_EQ(Y.shape()[0], 6);

    int expected = 1;
    for (size_t i = 0; i < 6; i++) {
        EXPECT_EQ(Y[i], expected++) << "Mismatch at Y[" << i << "]";
    }

    Y[0] = 99;
    EXPECT_EQ(X[0][0], 99); // check view semantics
}


TEST(TestTensorFlatten, TestFlattenInnerDims) {
    Tensor X(float32, {2, 3, 4}); 
    X.initialize();

    int val = 1;
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            for (size_t k = 0; k < 4; k++)
                X[i][j][k] = val++;

    Tensor Y = flatten(X, 1, -1); // collapse dims (1,2) → shape (2, 12)

    ASSERT_EQ(Y.shape().rank(), 2);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 12);

    int expected = 1;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 12; j++) {
            EXPECT_EQ(Y[i][j], expected++) 
                << "Mismatch at Y[" << i << "][" << j << "]";
        }
    }

    Y[1][5] = 123;
    EXPECT_EQ(X[1][1][1], 123); // verify mapping back
}


TEST(TestTensorFlatten, TestFlattenMiddleOnly) {
    Tensor X(float32, {2, 3, 4, 5}); 
    X.initialize();

    int val = 1;
    for (size_t a = 0; a < 2; a++)
        for (size_t b = 0; b < 3; b++)
            for (size_t c = 0; c < 4; c++)
                for (size_t d = 0; d < 5; d++)
                    X[a][b][c][d] = val++;

    Tensor Y = flatten(X, 1, 2); // collapse dims (1,2) → shape (2, 12, 5)

    ASSERT_EQ(Y.shape().rank(), 3);
    EXPECT_EQ(Y.shape()[0], 2);
    EXPECT_EQ(Y.shape()[1], 12);
    EXPECT_EQ(Y.shape()[2], 5);
 
    EXPECT_EQ(Y[0][0][0], 1);
 
    int expected_val = ((0 * 3 * 4 * 5) + (2 * 4 * 5) + (3 * 5) + 4 + 1);             
    EXPECT_EQ(Y[0][11][4], expected_val);  
}


TEST(TestTensorFlatten, TestFlattenInvalid) {
    Tensor X(float32, {2, 3, 4});
    X.initialize();
 
    EXPECT_THROW({
        auto Y = flatten(X, 2, 1);
    }, Exception);
 
    EXPECT_THROW({
        auto Y = flatten(X, 0, 10);
    }, Exception);
}


TEST(TestViews, TestAssignTensorToSlice) { 
    Tensor X(float32, {8, 3, 2}); 
    X.initialize({
        { {1.0000f, 0.0000f}, {1.0000f, 0.0000f}, {1.0000f, 0.0000f} },
        { {0.5403f, 0.8415f}, {0.9989f, 0.0464f}, {1.0000f, 0.0022f} },
        { {-0.4161f, 0.9093f}, {0.9957f, 0.0927f}, {1.0000f, 0.0043f} },
        { {-0.9900f, 0.1411f}, {0.9903f, 0.1388f}, {1.0000f, 0.0065f} },
        { {-0.6536f, -0.7568f}, {0.9828f, 0.1846f}, {1.0000f, 0.0086f} },
        { {0.2837f, -0.9589f}, {0.9732f, 0.2300f}, {0.9999f, 0.0108f} },
        { {0.9602f, -0.2794f}, {0.9615f, 0.2749f}, {0.9999f, 0.0129f} },
        { {0.7539f, 0.6570f}, {0.9477f, 0.3192f}, {0.9999f, 0.0151f} }
    }); 
     
    X[{2, 5}] = ones(float32, {3, 3, 2}); 

    Tensor X_expected(float32, {8, 3, 2}); X_expected.initialize({
        { {1.0000f, 0.0000f}, {1.0000f, 0.0000f}, {1.0000f, 0.0000f} },
        { {0.5403f, 0.8415f}, {0.9989f, 0.0464f}, {1.0000f, 0.0022f} },
        { {1.0000f, 1.0000f}, {1.0000f, 1.0000f}, {1.0000f, 1.0000f} },
        { {1.0000f, 1.0000f}, {1.0000f, 1.0000f}, {1.0000f, 1.0000f} },
        { {1.0000f, 1.0000f}, {1.0000f, 1.0000f}, {1.0000f, 1.0000f} },
        { {0.2837f, -0.9589f}, {0.9732f, 0.2300f}, {0.9999f, 0.0108f} },
        { {0.9602f, -0.2794f}, {0.9615f, 0.2749f}, {0.9999f, 0.0129f} },
        { {0.7539f, 0.6570f}, {0.9477f, 0.3192f}, {0.9999f, 0.0151f} }
    });

    EXPECT_TRUE(allclose(X, X_expected));
}


TEST(TestViews, Test4D5DView) {
 
    Tensor X = {{{{-0.6634, -0.1166, -0.3332,  0.7199},
          { 0.1709,  0.3820, -0.5651, -1.2078},
          {-0.9001, -0.4248, -0.4808,  0.2241},
          {-0.7889, -1.6041, -0.5045, -1.0839}},

         {{ 0.1870,  0.4933,  0.8438,  1.4492},
          { 0.7255,  0.0407, -0.8850, -0.4389},
          { 0.2041,  0.2771,  0.5841,  0.3641},
          { 0.8645,  0.5683, -0.2907,  0.3367}},

         {{-0.1537, -0.7270,  0.4192, -0.3193},
          {-0.4502,  0.6068,  0.2704, -0.9069},
          {-0.3498, -0.6477, -0.0190,  0.4983},
          {-1.2672, -0.2713, -0.6572,  0.0768}},

         {{-0.4298, -0.2925, -0.6602,  0.8198},
          {-0.2983,  0.9117,  0.4849,  0.1724},
          { 0.2354, -0.2089, -0.0449,  0.5041},
          {-1.1677,  0.4997, -0.0966, -0.4553}}},


        {{{ 0.5904, -0.3835,  0.1341, -0.8212},
          { 0.0073, -0.2007,  0.2516, -0.4214},
          {-1.1230, -1.1515,  0.1598,  0.1436},
          {-0.1146,  0.0072, -0.0412,  0.5098}},

         {{-0.6552,  0.0369, -0.9192, -0.4135},
          { 1.0371,  0.1792,  0.3646, -0.5280},
          {-0.4267,  0.6967,  0.4495,  1.0124},
          { 0.5513,  0.4260,  0.3625,  1.2069}},

         {{-0.0401,  0.4680, -0.3843, -0.3047},
          { 1.0414,  0.3671, -0.0839, -0.5699},
          {-0.6338, -0.2520, -0.9336, -0.1332},
          {-0.1550, -0.2921,  0.7204, -0.6646}},

         {{-0.0495,  0.5032,  0.6021, -0.3927},
          {-0.1664,  0.2872, -1.6430, -1.0185},
          {-0.1811, -0.8535, -1.1717, -0.4392},
          { 0.6354, -0.0745,  0.1813,  0.6837}}}};

    
    size_t batch_size = 2;
    size_t number_of_heads = 4;
    size_t sequence_length = 4;
    size_t heads_dimension = 4;

    Tensor Y = X.view(batch_size, number_of_heads, sequence_length, heads_dimension / 2, 2);

    Tensor Y_expected(float64, {2,4,4,2,2}); Y_expected.initialize({{{{{-0.6634, -0.1166},
            {-0.3332,  0.7199}},

           {{ 0.1709,  0.3820},
            {-0.5651, -1.2078}},

           {{-0.9001, -0.4248},
            {-0.4808,  0.2241}},

           {{-0.7889, -1.6041},
            {-0.5045, -1.0839}}},


          {{{ 0.1870,  0.4933},
            { 0.8438,  1.4492}},

           {{ 0.7255,  0.0407},
            {-0.8850, -0.4389}},

           {{ 0.2041,  0.2771},
            { 0.5841,  0.3641}},

           {{ 0.8645,  0.5683},
            {-0.2907,  0.3367}}},


          {{{-0.1537, -0.7270},
            { 0.4192, -0.3193}},

           {{-0.4502,  0.6068},
            { 0.2704, -0.9069}},

           {{-0.3498, -0.6477},
            {-0.0190,  0.4983}},

           {{-1.2672, -0.2713},
            {-0.6572,  0.0768}}},


          {{{-0.4298, -0.2925},
            {-0.6602,  0.8198}},

           {{-0.2983,  0.9117},
            { 0.4849,  0.1724}},

           {{ 0.2354, -0.2089},
            {-0.0449,  0.5041}},

           {{-1.1677,  0.4997},
            {-0.0966, -0.4553}}}},



         {{{{ 0.5904, -0.3835},
            { 0.1341, -0.8212}},

           {{ 0.0073, -0.2007},
            { 0.2516, -0.4214}},

           {{-1.1230, -1.1515},
            { 0.1598,  0.1436}},

           {{-0.1146,  0.0072},
            {-0.0412,  0.5098}}},


          {{{-0.6552,  0.0369},
            {-0.9192, -0.4135}},

           {{ 1.0371,  0.1792},
            { 0.3646, -0.5280}},

           {{-0.4267,  0.6967},
            { 0.4495,  1.0124}},

           {{ 0.5513,  0.4260},
            { 0.3625,  1.2069}}},


          {{{-0.0401,  0.4680},
            {-0.3843, -0.3047}},

           {{ 1.0414,  0.3671},
            {-0.0839, -0.5699}},

           {{-0.6338, -0.2520},
            {-0.9336, -0.1332}},

           {{-0.1550, -0.2921},
            { 0.7204, -0.6646}}},


          {{{-0.0495,  0.5032},
            { 0.6021, -0.3927}},

           {{-0.1664,  0.2872},
            {-1.6430, -1.0185}},

           {{-0.1811, -0.8535},
            {-1.1717, -0.4392}},

           {{ 0.6354, -0.0745},
            { 0.1813,  0.6837}}}}});
 
    EXPECT_TRUE(allclose(Y,  Y_expected));
}

/*

TEST(TestVIews, TestTransposedFirst) {
 
    Tensor X(float32, {2,4,4,4}); X.initialize({ 
      {{  {-0.6634, -0.1166, -0.3332,  0.7199},
          { 0.1870,  0.4933,  0.8438,  1.4492},
          {-0.1537, -0.7270,  0.4192, -0.3193},
          {-0.4298, -0.2925, -0.6602,  0.8198}},

      {{ 0.1709,  0.3820, -0.5651, -1.2078},
          { 0.7255,  0.0407, -0.8850, -0.4389},
          {-0.4502,  0.6068,  0.2704, -0.9069},
          {-0.2983,  0.9117,  0.4849,  0.1724}},

      {{-0.9001, -0.4248, -0.4808,  0.2241},
          { 0.2041,  0.2771,  0.5841,  0.3641},
          {-0.3498, -0.6477, -0.0190,  0.4983},
          { 0.2354, -0.2089, -0.0449,  0.5041}},

      {{-0.7889, -1.6041, -0.5045, -1.0839},
          { 0.8645,  0.5683, -0.2907,  0.3367},
          {-1.2672, -0.2713, -0.6572,  0.0768},
          {-1.1677,  0.4997, -0.0966, -0.4553}}},
 
      {{{ 0.5904, -0.3835,  0.1341, -0.8212},
          {-0.6552,  0.0369, -0.9192, -0.4135},
          {-0.0401,  0.4680, -0.3843, -0.3047},
          {-0.0495,  0.5032,  0.6021, -0.3927}},

      {{ 0.0073, -0.2007,  0.2516, -0.4214},
          { 1.0371,  0.1792,  0.3646, -0.5280},
          { 1.0414,  0.3671, -0.0839, -0.5699},
          {-0.1664,  0.2872, -1.6430, -1.0185}},

      {{-1.1230, -1.1515,  0.1598,  0.1436},
          {-0.4267,  0.6967,  0.4495,  1.0124},
          {-0.6338, -0.2520, -0.9336, -0.1332},
          {-0.1811, -0.8535, -1.1717, -0.4392}},

      {{-0.1146,  0.0072, -0.0412,  0.5098},
          { 0.5513,  0.4260,  0.3625,  1.2069},
          {-0.1550, -0.2921,  0.7204, -0.6646},
          { 0.6354, -0.0745,  0.1813,  0.6837}}}
      });
      
    X = X.transpose(1,2);

    size_t batch_size = 2;
    size_t number_of_heads = 4;
    size_t sequence_length = 4;
    size_t heads_dimension = 4;

    Tensor Y = X.view(batch_size, number_of_heads, sequence_length, heads_dimension / 2, 2); 
    
    Tensor Y_expected(float32, {2,4,4,2,2}); Y_expected.initialize( 
        {{{{{-0.6634, -0.1166},
          {-0.3332,  0.7199}},

          {{ 0.1709,  0.3820},
          {-0.5651, -1.2078}},

          {{-0.9001, -0.4248},
          {-0.4808,  0.2241}},

          {{-0.7889, -1.6041},
          {-0.5045, -1.0839}}},


        {{{ 0.1870,  0.4933},
          { 0.8438,  1.4492}},

          {{ 0.7255,  0.0407},
          {-0.8850, -0.4389}},

          {{ 0.2041,  0.2771},
          { 0.5841,  0.3641}},

          {{ 0.8645,  0.5683},
          {-0.2907,  0.3367}}},


        {{{-0.1537, -0.7270},
          { 0.4192, -0.3193}},

          {{-0.4502,  0.6068},
          { 0.2704, -0.9069}},

          {{-0.3498, -0.6477},
          {-0.0190,  0.4983}},

          {{-1.2672, -0.2713},
          {-0.6572,  0.0768}}},


        {{{-0.4298, -0.2925},
          {-0.6602,  0.8198}},

          {{-0.2983,  0.9117},
          { 0.4849,  0.1724}},

          {{ 0.2354, -0.2089},
          {-0.0449,  0.5041}},

          {{-1.1677,  0.4997},
          {-0.0966, -0.4553}}}},



        {{{{ 0.5904, -0.3835},
          { 0.1341, -0.8212}},

          {{ 0.0073, -0.2007},
          { 0.2516, -0.4214}},

          {{-1.1230, -1.1515},
          { 0.1598,  0.1436}},

          {{-0.1146,  0.0072},
          {-0.0412,  0.5098}}},


        {{{-0.6552,  0.0369},
          {-0.9192, -0.4135}},

          {{ 1.0371,  0.1792},
          { 0.3646, -0.5280}},

          {{-0.4267,  0.6967},
          { 0.4495,  1.0124}},

          {{ 0.5513,  0.4260},
          { 0.3625,  1.2069}}},


        {{{-0.0401,  0.4680},
          {-0.3843, -0.3047}},

          {{ 1.0414,  0.3671},
          {-0.0839, -0.5699}},

          {{-0.6338, -0.2520},
          {-0.9336, -0.1332}},

          {{-0.1550, -0.2921},
          { 0.7204, -0.6646}}},


        {{{-0.0495,  0.5032},
          { 0.6021, -0.3927}},

          {{-0.1664,  0.2872},
          {-1.6430, -1.0185}},

          {{-0.1811, -0.8535},
          {-1.1717, -0.4392}},

          {{ 0.6354, -0.0745},
          { 0.1813,  0.6837}}}}}
    );

    EXPECT_TRUE(allclose(Y,  Y_expected));
}
*/