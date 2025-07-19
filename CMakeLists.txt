cmake_minimum_required(VERSION 3.28)

project(TANNIC 
    VERSION 1.0
    DESCRIPTION "Tannic tensor library"
    LANGUAGES CXX  
)
 
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
 
option(TANNIC_ENABLE_CUDA "Enable CUDA support" ON)
option(TANNIC_ENABLE_TESTS "Build tests" ON)
 
include(cmake/CMakeLists.txt) 
include(GNUInstallDirs)
