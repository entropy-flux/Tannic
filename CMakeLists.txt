cmake_minimum_required(VERSION 3.22)
project(TANNIC LANGUAGES CXX)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled.")
else()
    message(STATUS "CUDA not found. Building without CUDA support.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


include_directories(include/tannic)
include_directories(src)


file(GLOB_RECURSE SRC_FILES
    src/*.cpp
    src/*.cu
)


find_package(OpenBLAS QUIET)

if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS}")
    add_definitions(-DOPENBLAS) 
    include_directories(${OpenBLAS_INCLUDE_DIRS})
else()
    message(STATUS "OpenBLAS not found, continuing without it.")
endif()


if(SRC_FILES)
    add_library(tannic STATIC ${SRC_FILES})
    target_include_directories(tannic PUBLIC include)

    if(CUDA_ENABLED)
        set_target_properties(tannic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(tannic PUBLIC CUDA::cudart)
        target_compile_definitions(tannic PUBLIC CUDA)
    endif()

    if(OpenBLAS_FOUND)
        target_link_libraries(tannic PUBLIC ${OpenBLAS_LIBRARIES})
        target_compile_definitions(tannic PUBLIC BLAS)
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
    message(STATUS "Found system-installed GoogleTest")
endif()


file(GLOB TEST_FILES
    tests/*.cpp
    tests/cpu/*.cpp
    tests/cuda/*.cpp
)

if(TEST_FILES)
    enable_testing()
    add_executable(tests ${TEST_FILES})

    target_include_directories(tests PRIVATE include)
    target_link_libraries(tests PRIVATE tannic)

    if(CUDA_ENABLED)
        set_target_properties(tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 20
        )
        target_link_libraries(tests PRIVATE CUDA::cudart)
    endif()

    if(GTest_FOUND)
        target_link_libraries(tests PRIVATE GTest::gtest_main)
    else()
        target_link_libraries(tests PRIVATE gtest_main)
    endif()

    include(GoogleTest)
    gtest_discover_tests(tests)
endif()
