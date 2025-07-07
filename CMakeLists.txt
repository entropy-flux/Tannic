cmake_minimum_required(VERSION 3.22)
project(TENSORLIB LANGUAGES CXX CUDA)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 86)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)
include_directories(src)

file(GLOB_RECURSE SRC_FILES
    src/*.cpp
    src/*.cu
)

# Find OpenBLAS if available
find_package(OpenBLAS QUIET)

if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS} ${OpenBLAS_LIBRARIES}")
    add_definitions(-DOPENBLAS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
else()
    message(STATUS "OpenBLAS not found, continuing without it.")
endif()

if(SRC_FILES)
    add_library(tensorlib STATIC ${SRC_FILES})

    set_target_properties(tensorlib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_include_directories(tensorlib PUBLIC include)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tensorlib PUBLIC CUDA::cudart)

    if(CUDAToolkit_FOUND)
        target_compile_definitions(tensorlib PUBLIC CUDA)
    endif()

    # Link OpenBLAS if found
    if(OpenBLAS_FOUND)
        target_link_libraries(tensorlib PUBLIC ${OpenBLAS_LIBRARIES})
    endif()
endif()

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found, fetching and building with FetchContent")

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Found system installed GoogleTest")
endif()

file(GLOB TEST_FILES
    tests/*.cpp
    tests/cpu/*.cpp
)

if(TEST_FILES)
    enable_testing()
    add_executable(tests ${TEST_FILES})

    set_target_properties(tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
    )

    target_include_directories(tests PRIVATE include)
    target_link_libraries(tests PRIVATE tensorlib CUDA::cudart)

    if(GTest_FOUND)
        target_link_libraries(tests PRIVATE GTest::gtest_main)
    else()
        target_link_libraries(tests PRIVATE gtest_main)
    endif()

    include(GoogleTest)
    gtest_discover_tests(tests)
endif()
