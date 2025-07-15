#include <gtest/gtest.h>
#include "Tensor.hpp" 
#include <gtest/gtest.h>
#include "Tensor.hpp" 

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
    
    Tensor slice = tensor[1]; 
    ASSERT_EQ(slice.offset(), 8);
    ASSERT_EQ(slice.shape(), Shape(2));
    ASSERT_EQ(slice.strides(), Strides(1));
    ASSERT_EQ(slice.rank(), 1);

    auto view = slice[0];
    ASSERT_EQ(view.rank(), 0);
    ASSERT_EQ(view.offset(), 8); 

    ASSERT_EQ((slice[0] == 3), true);
}   
