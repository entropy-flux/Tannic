#include <gtest/gtest.h>
#include "Tensor.hpp"

TEST(Test, Indexing) { 
    Tensor x({1,2,3}, float32);
    x[0][0][0] = 1.f;
    x[0][0][1] = 2.f;
    x[0][0][2] = 3.f;
    x[0][1][0] = 4.f;
    x[0][1][1] = 5.f;
    x[0][1][2] = 6.f;

    Tensor y = x[0];
    ASSERT_EQ(y[0][0], 1.f);
    ASSERT_EQ(y[0][1], 2.f);
    ASSERT_EQ(y[0][2], 3.f);
    ASSERT_EQ(y[1][0], 4.f);
    ASSERT_EQ(y[1][1], 5.f);
    ASSERT_EQ(y[1][2], 6.f);

    Tensor z = y[1];
    ASSERT_EQ(z[0], 4.f);
    ASSERT_EQ(z[1], 5.f);
    ASSERT_EQ(z[2], 6.f);

    Tensor w = z[1];
    float w_value = w.item<float>();
    ASSERT_EQ(w_value, 5.f);

    
    Tensor tensor({2, 3}, float32);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; j++)
            tensor[i][j] = float(i + j + 1);    
}

TEST(Test, Transpose) {
    // Create a 2x3 tensor with known values
    Tensor x({2, 3}, float32);
    x[0][0] = 1.f; x[0][1] = 2.f; x[0][2] = 3.f;
    x[1][0] = 4.f; x[1][1] = 5.f; x[1][2] = 6.f; 

    Tensor t = x.transpose(0, 1);
 
    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 2);
 
    ASSERT_EQ(t[0][0], 1.f);
    ASSERT_EQ(t[0][1], 4.f);
    ASSERT_EQ(t[1][0], 2.f);
    ASSERT_EQ(t[1][1], 5.f);
    ASSERT_EQ(t[2][0], 3.f);
    ASSERT_EQ(t[2][1], 6.f);
}

TEST(Test, TransposeNegativeIndices) {
    Tensor x({2, 3}, float32);
    x[0][0] = 1.f; x[0][1] = 2.f; x[0][2] = 3.f;
    x[1][0] = 4.f; x[1][1] = 5.f; x[1][2] = 6.f;

    // Using negative indices -2 and -1, which correspond to 0 and 1 respectively
    Tensor t = x.transpose(-2, -1);

    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 2);

    ASSERT_EQ(t[0][0], 1.f);
    ASSERT_EQ(t[0][1], 4.f);
    ASSERT_EQ(t[1][0], 2.f);
    ASSERT_EQ(t[1][1], 5.f);
    ASSERT_EQ(t[2][0], 3.f);
    ASSERT_EQ(t[2][1], 6.f);
}

TEST(Test, Squeeze) { 
    Tensor x({1, 3, 1, 2}, float32);
     
    x[0][0][0][0] = 7.f;
    x[0][2][0][1] = 9.f;

    Tensor squeezed = x.squeeze();
 
    ASSERT_EQ(squeezed.rank(), 2);
    ASSERT_EQ(squeezed.shape()[0], 3);
    ASSERT_EQ(squeezed.shape()[1], 2);
 
    ASSERT_EQ(squeezed[0][0], 7.f);
    ASSERT_EQ(squeezed[2][1], 9.f);
 
    Tensor y({1,1,1}, float32);
    Tensor y_squeezed = y.squeeze();
 
    ASSERT_EQ(y_squeezed.rank(), 1);
    ASSERT_EQ(y_squeezed.shape()[0], 1);
}
 

TEST(Test, Unsqueeze) {
    // Start with shape {3, 2}
    Tensor x({3, 2}, float32);
    x[0][0] = 1.f; x[2][1] = 5.f;

    // Unsqueeze at dim 0 -> shape should become {1, 3, 2}
    Tensor u0 = x.unsqueeze(0);
    ASSERT_EQ(u0.rank(), 3);
    ASSERT_EQ(u0.shape()[0], 1);
    ASSERT_EQ(u0.shape()[1], 3);
    ASSERT_EQ(u0.shape()[2], 2);
    ASSERT_EQ(u0[0][0][0], 1.f);
    ASSERT_EQ(u0[0][2][1], 5.f);
    
    Tensor u2 = x.unsqueeze(2);
    
    ASSERT_EQ(u2.rank(), 3);
    ASSERT_EQ(u2.shape()[0], 3);
    ASSERT_EQ(u2.shape()[1], 2);
    ASSERT_EQ(u2.shape()[2], 1);
    ASSERT_EQ(u2[0][0][0], 1.f);
    ASSERT_EQ(u2[2][1][0], 5.f);
    // Unsqueeze at dim 2 -> shape should become {3, 2, 1}
}


TEST(Test, UnsqueezeNegativeIndices) {
    Tensor x({3, 2}, float32);
    x[0][0] = 1.f; x[2][1] = 5.f;

    // Unsqueeze at -3 (which is equivalent to 0 for a rank 2 tensor)
    Tensor u_neg3 = x.unsqueeze(-3);
    ASSERT_EQ(u_neg3.rank(), 3);
    ASSERT_EQ(u_neg3.shape()[0], 1);
    ASSERT_EQ(u_neg3.shape()[1], 3);
    ASSERT_EQ(u_neg3.shape()[2], 2);
    ASSERT_EQ(u_neg3[0][0][0], 1.f);
    ASSERT_EQ(u_neg3[0][2][1], 5.f);

    // Unsqueeze at -1 (which is equivalent to 2 for a rank 2 tensor)
    Tensor u_neg1 = x.unsqueeze(-1);
    ASSERT_EQ(u_neg1.rank(), 3);
    ASSERT_EQ(u_neg1.shape()[0], 3);
    ASSERT_EQ(u_neg1.shape()[1], 2);
    ASSERT_EQ(u_neg1.shape()[2], 1);
    ASSERT_EQ(u_neg1[0][0][0], 1.f);
    ASSERT_EQ(u_neg1[2][1][0], 5.f);
}

TEST(Test, UnsqueezeVariadic) {
    Tensor x({3, 2}, float32);
    x[0][0] = 1.f; x[2][1] = 5.f;

    // Unsqueeze at dims 0 and 2 -> shape should become {1, 3, 1, 2}
    Tensor u_multi = x.unsqueeze(0, 2);
    ASSERT_EQ(u_multi.rank(), 4);
    ASSERT_EQ(u_multi.shape()[0], 1);
    ASSERT_EQ(u_multi.shape()[1], 3);
    ASSERT_EQ(u_multi.shape()[2], 1);
    ASSERT_EQ(u_multi.shape()[3], 2);
    ASSERT_EQ(u_multi[0][0][0][0], 1.f);
    ASSERT_EQ(u_multi[0][2][0][1], 5.f);

    // Unsqueeze at dims -1 and -4 -> equivalent to inserting at 2 and 0
    Tensor u_neg_multi = x.unsqueeze(-1, -4);
    ASSERT_EQ(u_neg_multi.rank(), 4);
    ASSERT_EQ(u_neg_multi.shape()[0], 1);
    ASSERT_EQ(u_neg_multi.shape()[1], 3);
    ASSERT_EQ(u_neg_multi.shape()[2], 2);
    ASSERT_EQ(u_neg_multi.shape()[3], 1);
    ASSERT_EQ(u_neg_multi[0][0][0][0], 1.f);
    ASSERT_EQ(u_neg_multi[0][2][1][0], 5.f);
}
