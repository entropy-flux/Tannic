#include <gtest/gtest.h>
#include "Tensor.hpp"
 
TEST(TestTensor, TestTensorInitialization) { 
    Tensor tensor(float32, {2, 2}); tensor.initialize(); 
    
    tensor[0, 0] = 1;
    tensor[0, 1] = 2;
    tensor[1, 0] = 3;
    tensor[1, 1] = 4;  
     
    EXPECT_EQ((tensor[0,0] == 1), true); 
    EXPECT_EQ((tensor[0,1] == 2), true); 
    EXPECT_EQ((tensor[1,0] == 3), true); 
    EXPECT_EQ((tensor[1,1] == 4), true); 

    EXPECT_EQ((tensor[0,0] == 2), false); 
    EXPECT_EQ((tensor[0,1] == 4), false); 
    EXPECT_EQ((tensor[1,0] == 6), false); 
    EXPECT_EQ((tensor[1,1] == 8), false);
}