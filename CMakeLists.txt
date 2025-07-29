cmake_minimum_required(VERSION 3.28)

project(TANNIC 
    VERSION 1.0
    DESCRIPTION "Tannic tensor library"
    LANGUAGES CXX CUDA
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TANNIC_ENABLE_CUDA "Enable CUDA support" ON)
option(TANNIC_ENABLE_TESTS "Build tests" ON)
option(TANNIC_BUILD_MAIN "Build the main executable" ON)  

include(cmake/CMakeLists.txt) 

if(TANNIC_BUILD_MAIN)
    add_executable(main main.cpp)
    target_link_libraries(main PRIVATE tannic CUDA::cudart)
    set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(main PRIVATE include)
    target_compile_features(main PRIVATE cxx_std_23)
endif()
