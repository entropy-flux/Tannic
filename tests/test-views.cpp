#include <gtest/gtest.h>

#include "Parameter.hpp"
#include "Slices.hpp"

using namespace tannic;

TEST(TestView, TestSliceParam) { 
    constexpr Parameter tensor(float32, {3,1,4,5}); 
    constexpr auto view = tensor[2][0][{1,3}];  
    static_assert(view.shape() == Shape(2,5));   
    static_assert(view.strides() == Strides(5, 1));
    static_assert(view.offset() == 45*dsizeof(float32));    
}  

TEST(TestView, TestSlice2Param) {
    constexpr Parameter tensor(int8, {6,2,3,4,4});
    constexpr auto view = tensor[{1,3}][{0,2}][1]; 
    static_assert(view.shape() == Shape(2, 2, 4, 4)); 
    static_assert(view.strides() == Strides(96, 48, 4, 1)); 
    static_assert(view.offset() == 112 * dsizeof(int8));
}
 

TEST(TestView, TestNegativeSliceParam) {
    constexpr Parameter tensor(int8, {6,2,3,4,4});
    constexpr auto view = tensor[{-5,-4}][{0,-1}][-2]; 
    static_assert(view.shape()[0] == 2); 
    static_assert(view.shape()[1] == 2); 
    static_assert(view.shape() == Shape(2, 2, 4, 4)); 
    static_assert(view.strides() == Strides(96, 48, 4, 1));  
    static_assert(view.offset() == 112 * dsizeof(int8));
}

TEST(TestView, TestTransposeParam) {
    constexpr Parameter tensor(float32, {3, 4}); 
    constexpr auto view = tensor.transpose(0, 1);   
    static_assert(view.shape() == Shape(4, 3));
    static_assert(view.strides() == Strides(1, 4));   
    static_assert(view.offset() == 0 * dsizeof(float32));
}

TEST(TestView, TestTransposeHigherRankParam) {
    constexpr Parameter tensor(int8, {2, 3, 4}); 
    constexpr auto view = tensor.transpose(1, 2);   
    
    static_assert(view.shape() == Shape(2, 4, 3));
    static_assert(view.strides() == Strides(12, 1, 4));  
    static_assert(view.offset() == 0 * dsizeof(int8));
}

TEST(TestView, TestTransposeNonAdjacentParam) {
    constexpr Parameter tensor(float32, {2, 3, 4, 5}); 
    constexpr auto view = tensor.transpose(0, 3);   
    
    static_assert(view.shape() == Shape(5, 3, 4, 2));
    static_assert(view.strides() == Strides(1, 20, 5, 60));   
    static_assert(view.offset() == 0 * dsizeof(float32));
}

TEST(TestView, TestTransposeAfterSliceParam) {
    constexpr Parameter tensor(int8, {6, 2, 3, 4});  
    constexpr auto view = transpose(tensor[{1, 4}][0], 0, 2);  
    
    static_assert(view.shape() == Shape(4, 3, 3)); 
    static_assert(view.offset() == 24 * dsizeof(int8));   
}