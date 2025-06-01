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


add_library(operations 
    ${CMAKE_SOURCE_DIR}/../src/operations.cpp
    ${CMAKE_SOURCE_DIR}/../src/cpu/negation.cpp
    ${CMAKE_SOURCE_DIR}/../src/cpu/addition.cpp
    ${CMAKE_SOURCE_DIR}/../src/cpu/subtraction.cpp
    ${CMAKE_SOURCE_DIR}/../src/cpu/multiplication.cpp
    ${CMAKE_SOURCE_DIR}/../src/cpu/matmul.cpp 
)

add_executable(runTests_shape test_shape.cpp)  
add_executable(runTests_broadcast operations/test_broadcast.cpp) 
add_executable(runTests_negation operations/test_negation.cpp) 
add_executable(runTests_addition operations/test_addition.cpp) 
add_executable(runTests_subtraction operations/test_subtraction.cpp) 
add_executable(runTests_multiplication operations/test_multiplication.cpp) 
add_executable(runTests_matmul operations/test_matmul.cpp) 
add_executable(runTests_storage test_storage.cpp) 
add_executable(runTests_types test_types.cpp)
add_executable(runTests_tensor test_tensor.cpp) 
add_executable(runTests_serialization test_serialization.cpp) 

foreach(
    test_target 
    runTests_shape  
    runTests_broadcast  
    runTests_negation 
    runTests_addition
    runTests_subtraction
    runTests_multiplication
    runTests_matmul 
    runTests_storage
    runTests_types   
    runTests_tensor  
    runTests_serialization
)
    target_link_libraries(${test_target} 
        GTest::GTest 
        GTest::Main 
        operations
    )
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()