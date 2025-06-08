#include <gtest/gtest.h>
#include "Freelist.hpp"
#include "Resources.hpp"
#include "Pool.hpp"  

TEST(Test, Allocation) { 
    Host host{};
    void* memory = host.allocate(512);
    float value = 5;
    host.copy(memory, &value, sizeof(float));
    float returned = static_cast<float>(value);
    ASSERT_EQ(value, 5); 
}
 
TEST(Test, CUDAAllocation) { 
    Device device(0);
    void* memory = device.allocate(512);
    float value = 5;
    device.copy(memory, &value, sizeof(float));
    float returned = static_cast<float>(value);
    ASSERT_EQ(value, 5); 
}
 