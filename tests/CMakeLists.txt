cmake_minimum_required(VERSION 3.10)
project(TENSORLIB)

set(CMAKE_CXX_STANDARD 23)


include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../src
)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_compile_options(-fconstexpr-depth=512)
 

add_library(naive_impl 
    ${CMAKE_SOURCE_DIR}/../src/cpu/naive.cpp
)

add_executable(runTests_types test_types.cpp)
add_executable(runTests_shape test_shape.cpp) 
add_executable(runTests_tensor test_tensor.cpp)
add_executable(runTests_expressions test_expressions.cpp)
add_executable(runTests_parameter test_parameter.cpp)
 
foreach(test_target 
    runTests_types 
    runTests_shape 
    runTests_tensor 
    runTests_expressions
    runTests_parameter
)
    target_link_libraries(${test_target} 
        GTest::GTest 
        GTest::Main 
        naive_impl
    )
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()