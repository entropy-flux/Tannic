#include <gtest/gtest.h>
#include "Tensor.hpp"  

using namespace tannic;

TEST(TestView, TestSlice) { 
    Tensor tensor(int32, {2,2});  tensor.initialize();
    tensor[0,0] = 1;
    tensor[0,1] = 2;
    tensor[1,0] = 3;
    tensor[1,1] = 4;
  
    ASSERT_EQ((tensor[0,0] == 1), true); // Note: gtest macros don't support c++23 variadic arguments in operator[]
    ASSERT_EQ((tensor[0,1] == 2), true);
    ASSERT_EQ((tensor[1,0] == 3), true);
    ASSERT_EQ((tensor[1,1] == 4), true);
 
    ASSERT_EQ(tensor[0][0], 1); // This is supported.
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