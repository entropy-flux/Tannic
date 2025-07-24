cmake_minimum_required(VERSION 3.28)
project(TANNIC_TESTS LANGUAGES CXX)

set(TANNIC_BUILD_DIR "${CMAKE_SOURCE_DIR}/../build" CACHE PATH "Path to TANNIC build directory")
 
find_library(TANNIC_LIBRARY
    NAMES tannic
    PATHS ${TANNIC_BUILD_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)
 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
 
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
    add_definitions(-DOPENBLAS)
else()
    message(STATUS "OpenBLAS not found, continuing without it.")
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
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Found system-installed GoogleTest")
endif()
 
file(GLOB TEST_FILES
    "*.cpp"
    "cpu/*.cpp"
)

if(TEST_FILES)
    enable_testing()
    add_executable(tannic_tests ${TEST_FILES})
 
    target_include_directories(tannic_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/../include/tannic"
        "${CMAKE_SOURCE_DIR}/../src"
    )
 
    target_link_libraries(tannic_tests PRIVATE ${TANNIC_LIBRARY})
 
    if(OpenBLAS_FOUND)
        target_link_libraries(tannic_tests PRIVATE ${OpenBLAS_LIBRARIES})
    endif()
 
    if(GTest_FOUND)
        target_link_libraries(tannic_tests PRIVATE GTest::GTest GTest::Main)
    else()
        target_link_libraries(tannic_tests PRIVATE gtest gtest_main)
    endif()

    include(GoogleTest)
    gtest_discover_tests(tannic_tests)
endif()