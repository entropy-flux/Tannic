#include <gtest/gtest.h>
#include "Tensor.hpp" 
#include "Algebra/Transformations.hpp"

TEST(Test, MatrixMatmulFloat) {
    Tensor x({3,4}, float32);
    Tensor y({4,5}, float32);
 
    x[0][0] = 4.f; x[0][1] = 2.f; x[0][2] = 6.f; x[0][3] = 0.f;
    x[1][0] = 1.f; x[1][1] = 3.f; x[1][2] = 5.f; x[1][3] = 7.f;
    x[2][0] = 2.f; x[2][1] = 4.f; x[2][2] = 6.f; x[2][3] = 8.f;
 
    y[0][0] = 1.f; y[0][1] = 2.f; y[0][2] = 3.f; y[0][3] = 4.f; y[0][4] = 5.f;
    y[1][0] = 0.f; y[1][1] = 1.f; y[1][2] = 0.f; y[1][3] = 1.f; y[1][4] = 0.f;
    y[2][0] = 2.f; y[2][1] = 3.f; y[2][2] = 4.f; y[2][3] = 5.f; y[2][4] = 6.f;
    y[3][0] = 1.f; y[3][1] = 0.f; y[3][2] = 1.f; y[3][3] = 0.f; y[3][4] = 1.f;
 
    Tensor z = matmul(x, y);

    EXPECT_NEAR(z[0][0].item<float>(), 16.f, 1e-5);
    EXPECT_NEAR(z[0][1].item<float>(), 28.f, 1e-5);
    EXPECT_NEAR(z[0][2].item<float>(), 36.f, 1e-5);
    EXPECT_NEAR(z[0][3].item<float>(), 48.f, 1e-5);
    EXPECT_NEAR(z[0][4].item<float>(), 56.f, 1e-5);

    EXPECT_NEAR(z[1][0].item<float>(), 18.f, 1e-5);
    EXPECT_NEAR(z[1][1].item<float>(), 20.f, 1e-5);
    EXPECT_NEAR(z[1][2].item<float>(), 30.f, 1e-5);
    EXPECT_NEAR(z[1][3].item<float>(), 32.f, 1e-5);
    EXPECT_NEAR(z[1][4].item<float>(), 42.f, 1e-5);

    EXPECT_NEAR(z[2][0].item<float>(), 22.f, 1e-5);
    EXPECT_NEAR(z[2][1].item<float>(), 26.f, 1e-5);
    EXPECT_NEAR(z[2][2].item<float>(), 38.f, 1e-5);
    EXPECT_NEAR(z[2][3].item<float>(), 42.f, 1e-5);
    EXPECT_NEAR(z[2][4].item<float>(), 54.f, 1e-5);
}



TEST(Test, MatrixMatmulFloatDouble) {
    Tensor x({3, 4}, float32);  
    Tensor y({4, 5}, float64); 

    x[0][0] = 4.f; x[0][1] = 2.f; x[0][2] = 6.f; x[0][3] = 0.f;
    x[1][0] = 1.f; x[1][1] = 3.f; x[1][2] = 5.f; x[1][3] = 7.f;
    x[2][0] = 2.f; x[2][1] = 4.f; x[2][2] = 6.f; x[2][3] = 8.f;

    y[0][0] = 1.0; y[0][1] = 2.0; y[0][2] = 3.0; y[0][3] = 4.0; y[0][4] = 5.0;
    y[1][0] = 0.0; y[1][1] = 1.0; y[1][2] = 0.0; y[1][3] = 1.0; y[1][4] = 0.0;
    y[2][0] = 2.0; y[2][1] = 3.0; y[2][2] = 4.0; y[2][3] = 5.0; y[2][4] = 6.0;
    y[3][0] = 1.0; y[3][1] = 0.0; y[3][2] = 1.0; y[3][3] = 0.0; y[3][4] = 1.0;

    Tensor z = matmul(x, y);  // should infer output dtype or promote internally

    EXPECT_NEAR(z[0][0].item<double>(), 16, 1e-5);
    EXPECT_NEAR(z[0][1].item<double>(), 28, 1e-5);
    EXPECT_NEAR(z[0][2].item<double>(), 36, 1e-5);
    EXPECT_NEAR(z[0][3].item<double>(), 48, 1e-5);
    EXPECT_NEAR(z[0][4].item<double>(), 56, 1e-5);

    EXPECT_NEAR(z[1][0].item<double>(), 18, 1e-5);
    EXPECT_NEAR(z[1][1].item<double>(), 20, 1e-5);
    EXPECT_NEAR(z[1][2].item<double>(), 30, 1e-5);
    EXPECT_NEAR(z[1][3].item<double>(), 32, 1e-5);
    EXPECT_NEAR(z[1][4].item<double>(), 42, 1e-5);

    EXPECT_NEAR(z[2][0].item<double>(), 22, 1e-5);
    EXPECT_NEAR(z[2][1].item<double>(), 26, 1e-5);
    EXPECT_NEAR(z[2][2].item<double>(), 38, 1e-5);
    EXPECT_NEAR(z[2][3].item<double>(), 42, 1e-5);
    EXPECT_NEAR(z[2][4].item<double>(), 54, 1e-5);
}


TEST(Test, MatrixMatmulTransposedMultiplier) {
    Tensor x({3,4}, float32);
    Tensor y({5,4}, float32); 
 
    x[0][0] = 4.f; x[0][1] = 2.f; x[0][2] = 6.f; x[0][3] = 0.f;
    x[1][0] = 1.f; x[1][1] = 3.f; x[1][2] = 5.f; x[1][3] = 7.f;
    x[2][0] = 2.f; x[2][1] = 4.f; x[2][2] = 6.f; x[2][3] = 8.f;
 
    y[0][0] = 1.f; y[0][1] = 0.f; y[0][2] = 2.f; y[0][3] = 1.f;
    y[1][0] = 2.f; y[1][1] = 1.f; y[1][2] = 3.f; y[1][3] = 0.f;
    y[2][0] = 3.f; y[2][1] = 0.f; y[2][2] = 4.f; y[2][3] = 1.f;
    y[3][0] = 4.f; y[3][1] = 1.f; y[3][2] = 5.f; y[3][3] = 0.f;
    y[4][0] = 5.f; y[4][1] = 0.f; y[4][2] = 6.f; y[4][3] = 1.f;

    Tensor z = matmul(x, y.transpose(-1,-2));
    EXPECT_NEAR(z[0][0].item<float>(), 16.f, 1e-5);
    EXPECT_NEAR(z[0][1].item<float>(), 28.f, 1e-5);
    EXPECT_NEAR(z[0][2].item<float>(), 36.f, 1e-5);
    EXPECT_NEAR(z[0][3].item<float>(), 48.f, 1e-5);
    EXPECT_NEAR(z[0][4].item<float>(), 56.f, 1e-5);

    EXPECT_NEAR(z[1][0].item<float>(), 18.f, 1e-5);
    EXPECT_NEAR(z[1][1].item<float>(), 20.f, 1e-5);
    EXPECT_NEAR(z[1][2].item<float>(), 30.f, 1e-5);
    EXPECT_NEAR(z[1][3].item<float>(), 32.f, 1e-5);
    EXPECT_NEAR(z[1][4].item<float>(), 42.f, 1e-5);

    EXPECT_NEAR(z[2][0].item<float>(), 22.f, 1e-5);
    EXPECT_NEAR(z[2][1].item<float>(), 26.f, 1e-5);
    EXPECT_NEAR(z[2][2].item<float>(), 38.f, 1e-5);
    EXPECT_NEAR(z[2][3].item<float>(), 42.f, 1e-5);
    EXPECT_NEAR(z[2][4].item<float>(), 54.f, 1e-5);
}
