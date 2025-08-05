cmake_minimum_required(VERSION 3.28)

project(TANNIC LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled.")
    find_package(CUDAToolkit)
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA not found. Building without CUDA support.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TANNIC_ENABLE_CUDA "Enable CUDA support" ON)
option(TANNIC_ENABLE_TESTS "Build tests" ON)
option(TANNIC_BUILD_MAIN "Build the main executable" ON)  

include(cmake/CMakeLists.txt) 

if(TANNIC_BUILD_MAIN)
    add_executable(main main.cpp)
    if(CUDA_ENABLED)
        target_link_libraries(main PRIVATE CUDA::cudart)
        set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    target_include_directories(main PRIVATE include)
    target_compile_features(main PRIVATE cxx_std_23)
    
    # Link the tannic library here
    target_link_libraries(main PRIVATE tannic)
endif()
