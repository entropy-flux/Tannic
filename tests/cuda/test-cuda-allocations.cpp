#ifdef CUDA
#include <gtest/gtest.h> 
#include "resources.hpp"
 
using namespace tannic; 
TEST(TestCUDADevice, TestCUDAAlloc) {
    Device device(0);
    ASSERT_EQ(device.id(), 0); 
    void* ptr1 = device.allocate(256); 
    void* ptr2 = device.allocate(256);
    void* ptr3 = device.allocate(256);
    void* ptr4 = device.allocate(256);
    void* ptr5 = device.allocate(256); 
    device.deallocate(ptr1, 256);
    device.deallocate(ptr2, 256);
    device.deallocate(ptr3, 256);
    device.deallocate(ptr4, 256);
    device.deallocate(ptr5, 256);
}

#endif