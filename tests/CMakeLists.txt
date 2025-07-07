cmake_minimum_required(VERSION 3.28)
project(TENSORLIB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
 
include_directories(include src) 
 
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)
 
add_library(tensorlib STATIC ${SOURCES})
 
set_target_properties(tensorlib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
 
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
 
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
 
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    tests/*.cpp
    tests/cpu/*.cpp
)
 
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE tensorlib GTest::gtest_main)
 
target_include_directories(tests PRIVATE include)
 
include(GoogleTest)
gtest_discover_tests(tests)