#include <gtest/gtest.h>
#include "tensor.hpp"  

using namespace tannic;

TEST(TestView, TestSlice) { 
    Tensor tensor(int32, {2,2});  tensor.initialize();
    tensor[0,0] = 1;
    tensor[0,1] = 2;
    tensor[1,0] = 3;
    tensor[1,1] = 4;
  
    ASSERT_EQ((tensor[0,0] == 1), true);  
    ASSERT_EQ((tensor[0,1] == 2), true);
    ASSERT_EQ((tensor[1,0] == 3), true);
    ASSERT_EQ((tensor[1,1] == 4), true);
 
    ASSERT_EQ(tensor[0][0], 1); 
    ASSERT_EQ(tensor[0][1], 2);
    ASSERT_EQ(tensor[1][0], 3);
    ASSERT_EQ(tensor[1][1], 4); 
    
    Tensor slice = tensor[1]; 
    ASSERT_EQ(slice.offset(), 8);
    ASSERT_EQ(slice.shape(), Shape(2));
    ASSERT_EQ(slice.strides(), Strides(1));
    ASSERT_EQ(slice.rank(), 1);

    auto view = slice[0];
    ASSERT_EQ(view.rank(), 0);  
    ASSERT_EQ((slice[0] == 3), true);
}    


TEST(TestTensor, TestInitializerLists) {  
    Tensor X = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(X.rank(), 1);
    ASSERT_EQ(X.shape(), Shape({4}));
    ASSERT_EQ(X[0], 1.0f);
    ASSERT_EQ(X[3], 4.0f);
    
    Tensor Y = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    ASSERT_EQ(Y.rank(), 2);
    ASSERT_EQ(Y.shape(), Shape({2,3}));
    ASSERT_EQ(Y[0][0], 1.0f);
    ASSERT_EQ(Y[0][2], 3.0f);
    ASSERT_EQ(Y[1][0], 4.0f);
    ASSERT_EQ(Y[1][2], 6.0f);
  
    Tensor Z = {
        {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        },
        {
            {5.0f, 6.0f},
            {7.0f, 8.0f}
        }
    };
    ASSERT_EQ(Z.rank(), 3);
    ASSERT_EQ(Z.shape(), Shape({2,2,2}));
    ASSERT_EQ(Z[0][0][0], 1.0f);
    ASSERT_EQ(Z[0][1][1], 4.0f);
    ASSERT_EQ(Z[1][0][1], 6.0f);
    ASSERT_EQ(Z[1][1][1], 8.0f); 
}

