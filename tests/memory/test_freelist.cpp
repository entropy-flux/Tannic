#include <gtest/gtest.h>
#include "Freelist.hpp"
#include "Resources.hpp"
#include "Pool.hpp" 
 
class Test : public ::testing::Test {
protected:    
    Pool<Host, Freelist> pool; 
    Test() : pool(Host{}) {
        pool.reserve(512*B);
    }
};

TEST_F(Test, Prefix) {     
    ASSERT_EQ(1*B, 1);
    ASSERT_EQ(1*KB, 1024);
    ASSERT_EQ(1*MB, 1024*1024);
    ASSERT_EQ(1*GB, 1024*1024*1024);
}

TEST_F(Test, Order) {
    ASSERT_EQ(Freelist::order(1), 0);
    ASSERT_EQ(Freelist::order(2), 1);
    ASSERT_EQ(Freelist::order(3), 2);
    ASSERT_EQ(Freelist::order(4), 2);
    ASSERT_EQ(Freelist::order(7), 3);
    ASSERT_EQ(Freelist::order(8), 3);
}

TEST_F(Test, Allocation) {  
    auto allocator = pool.allocator();

    void* address = allocator.allocate(sizeof(float)*8);   
    float* array = static_cast<float*>(address);
    for(auto index = 0; index < 8; index++) {
        array[index] = float(index);
    }
    ASSERT_EQ(array[0], 0.0f);
    ASSERT_EQ(array[1], 1.0f);
    ASSERT_EQ(array[7], 7.0f);
    ASSERT_EQ(allocator.available(), 512 - allocator.header - (sizeof(float)*8 + allocator.header));
} 

TEST_F(Test, Reusage) { 
    auto allocator = pool.allocator();

    void* address = allocator.allocate(sizeof(float)*8);   
    float* array = static_cast<float*>(address);
    for(auto index = 0; index < 8; index++) {
        array[index] = float(index);
    }

    allocator.deallocate(address, 0);
    void* different_order_array = allocator.reuse(sizeof(float)*9);
    ASSERT_EQ(different_order_array, nullptr);

    void* recycled_address = allocator.reuse(sizeof(float)*7);
    float* recycled_array = static_cast<float*>(address);
    ASSERT_EQ(recycled_array[0], 0.0f);
    ASSERT_EQ(recycled_array[1], 1.0f);
    ASSERT_EQ(recycled_array[6], 6.0f);
}