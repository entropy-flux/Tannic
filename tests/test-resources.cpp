#include <gtest/gtest.h>
#include "Resources.hpp"

TEST(Resources, Host) { 
    Host host;
    std::size_t size = 8;

    char* dst = static_cast<char*>(host.allocate(size));
    ASSERT_NE(dst, nullptr); 

    host.deallocate(dst, size);
}

#ifdef CUDA

TEST(Resources, PinnedHost) {
    Host host(true);
    std::size_t size = 8;

    char* dst = static_cast<char*>(host.allocate(size));
    ASSERT_NE(dst, nullptr);

    host.deallocate(dst, size);
}

TEST(Resources, Device) {
    Device device{};
    std::size_t size = 8;

    char* dst = static_cast<char*>(device.allocate(size));
    ASSERT_NE(dst, nullptr);

    device.deallocate(dst, size);
}

#else

TEST(Resources, PinnedHost) { 
    Host host(true);
    std::size_t size = 8;

    EXPECT_THROW({
        host.allocate(size);
    }, std::runtime_error);
}

TEST(Resources, Device) { 
    EXPECT_THROW({
        Device device;
    }, std::runtime_error);
}

#endif