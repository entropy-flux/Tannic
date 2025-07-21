#include <gtest/gtest.h>
#include "Tensor.hpp"
#include "Serialization.hpp"

using namespace tannic;
 
TEST(Test, Serialization) {      
    Tensor tensor(float32, {2,4,3}); tensor.initialize();
    tensor[{0,-1}] = 1;
    Blob serialized = serialize(tensor);
    write(serialized, "tensor.tannic");
    Blob readed = read("tensor.tannic");
    Tensor deserialized = deserialize(readed);
    ASSERT_EQ(deserialized.shape(), Shape(2,4,3));
    ASSERT_EQ(deserialized.strides(), Strides(12,3,1)); 
    ASSERT_EQ(deserialized[0][0][0], 1);
    ASSERT_EQ(deserialized[1][3][2], 1);
}  