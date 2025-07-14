#include <gtest/gtest.h>
#include "Resources.hpp"
 
TEST(Resources, Host) { 
    Host host;
    std::size_t size = 8;

    char* dst = static_cast<char*>(host.allocate(size));
    ASSERT_NE(dst, nullptr); 

    host.deallocate(dst, size);
}

/*
#ifdef CUDA 

TEST(Resources, Device) {
    Device device{};
    std::size_t size = 8;

    char* dst = static_cast<char*>(device.allocate(size));
    ASSERT_NE(dst, nullptr);

    device.deallocate(dst, size);
}

#else

TEST(Resources, Device) { 
    EXPECT_THROW({
        Device device;
    }, std::runtime_error);
}

#endif 
*/