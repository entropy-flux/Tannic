#include <gtest/gtest.h>
#include "IO/Serialization.hpp"
#include "IO/Persistence.hpp"

TEST(Test, Metadata) { 
    Shape shape(1,2,3,4);
    Strides strides(1,2,3,4);
    Metadata metadata(float32, shape, strides); 
    
    for (auto index = 0; index < metadata.rank; ++index) 
        EXPECT_EQ(metadata.shape[index], shape[index]);
    
    for (auto index = 0; index < metadata.rank; ++index) 
        EXPECT_EQ(metadata.strides[index], strides[index]);
}


TEST(Test, Serialization) {
    Tensor tensor({2, 3}, float32);
    
    std::cout << tensor << std::endl;


    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; j++)
            tensor[i][j] = static_cast<float>(i + j + 1);    
 
    
    Blob serialized = serialize(tensor, 64);
    write(serialized, "tensor.dat");
    
    Blob retrieved = read("tensor.dat");
    Tensor deserialized = deserialize(retrieved);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; j++)
            ASSERT_EQ(deserialized[i][j], static_cast<float>(i + j + 1));
}
