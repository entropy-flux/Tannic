#ifdef CUDA
#include <gtest/gtest.h>
#include "tensor.hpp"  

using namespace tannic;

TEST(TestCUDAView, TestCUDASlice) { 
    Tensor tensor(int32, {2,2});  tensor.initialize(Device(0));
    tensor[0,0] = 1;
    tensor[0,1] = 2;
    tensor[1,0] = 3;
    tensor[1,1] = 4;
   
    ASSERT_EQ(tensor[0][0], 1);  
    ASSERT_EQ(tensor[0][1], 2);
    ASSERT_EQ(tensor[1][0], 3);
    ASSERT_EQ(tensor[1][1], 4); 
    
    Tensor slice = tensor[1]; 
    ASSERT_EQ(slice.offset(), 8);
    ASSERT_EQ(slice.shape(), Shape(2));
    ASSERT_EQ(slice.strides(), Strides(1));
    ASSERT_EQ(slice.rank(), 1);

    Tensor view = slice[0];
    ASSERT_EQ(view.rank(), 0);  
    ASSERT_EQ(view.shape(), Shape());  
    ASSERT_EQ(slice[0], 3);
}    

TEST(TestCUDAView, NegativeIndexing) {
    Tensor tensor(int32, {3});
    tensor.initialize(Device(0));
    tensor[0] = 10;
    tensor[1] = 20;
    tensor[2] = 30;
    ASSERT_EQ(tensor[-1], 30);
    ASSERT_EQ(tensor[-3], 10);
}  

#endif