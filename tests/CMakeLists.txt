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
 

#add_library(broadcasting 
#    ${CMAKE_SOURCE_DIR}/../src/broadcasting.cpp
#)

add_executable(runTests_types test_types.cpp)
add_executable(runTests_shape test_shape.cpp)  
add_executable(runTests_operations test_operations.cpp) 
add_executable(runTests_storage test_storage.cpp) 
add_executable(runTests_tensor test_tensor.cpp) 
 
 
foreach(test_target 
    runTests_types 
    runTests_shape  
    runTests_operations   
    runTests_storage  
    runTests_tensor  
)
    target_link_libraries(${test_target} 
        GTest::GTest 
        GTest::Main 
#        broadcasting
    )
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()