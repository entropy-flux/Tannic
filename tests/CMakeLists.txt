cmake_minimum_required(VERSION 3.10)
project(TENSORLIB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../src/memory.cu 
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../include/Tannic
    ${CMAKE_SOURCE_DIR}/../src
)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_compile_options(-fconstexpr-depth=512)
 
find_library(OPENBLAS_LIB openblas)
if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS library not found!")
endif()
 
add_library(algebra 
    ${CMAKE_SOURCE_DIR}/../src/algebra.cpp
    ${CMAKE_SOURCE_DIR}/../src/memory.cpp
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/neg.cpp
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/add.cpp
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/sub.cpp
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/mul.cpp
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/fun.cpp   
    ${CMAKE_SOURCE_DIR}/../src/kernels/cpu/matmul.cpp   
    ${CMAKE_SOURCE_DIR}/../src/kernels/openblas/scal.cpp  
    ${CMAKE_SOURCE_DIR}/../src/kernels/openblas/gemm.cpp  
    ${CMAKE_SOURCE_DIR}/../src/kernels/openblas/axpy.cpp  
    ${CMAKE_SOURCE_DIR}/../src/kernels/openblas/copy.cpp   
    ${CUDA_SOURCES}
)
 
target_link_libraries(algebra PRIVATE ${OPENBLAS_LIB})
 
set_target_properties(algebra PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(runTests_shape test_shape.cpp)  
add_executable(runTests_broadcast algebra/test_broadcast.cpp) 
add_executable(runTests_functions algebra/test_functions.cpp)
add_executable(runTests_negation algebra/test_negation.cpp) 
add_executable(runTests_addition algebra/test_addition.cpp) 
add_executable(runTests_subtraction algebra/test_subtraction.cpp) 
add_executable(runTests_multiplication algebra/test_multiplication.cpp) 
add_executable(runTests_linear algebra/test_linear.cpp)  
add_executable(runTests_storage test_storage.cpp) 
add_executable(runTests_types test_types.cpp)
add_executable(runTests_tensor test_tensor.cpp) 
add_executable(runTests_serialization test_serialization.cpp) 
add_executable(runTests_freelist memory/test_freelist.cpp) 
add_executable(runTests_modules test_modules.cpp) 

foreach(
    test_target 
    runTests_shape  
    runTests_broadcast 
    runTests_functions    
    runTests_negation 
    runTests_addition
    runTests_subtraction
    runTests_multiplication
    runTests_linear  
    runTests_storage
    runTests_types   
    runTests_tensor  
    runTests_serialization  
    runTests_freelist
    runTests_modules
)
    target_link_libraries(${test_target} 
        GTest::GTest 
        GTest::Main 
        algebra
    ) 
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()