cmake_minimum_required(VERSION 3.28)
project(TENSORLIB LANGUAGES CXX)

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
)

# Check if CUDA compiler is available, enable if yes
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA enabled.")
    # Add CUDA source files only if CUDA enabled
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        src/*.cu
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
else()
    message(STATUS "CUDA not found, building without CUDA.")
endif()

add_library(tensorlib STATIC ${SOURCES})

if(CUDA_ENABLED)
    set_target_properties(tensorlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

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