#include <gtest/gtest.h>

#include "Parameter.hpp"
#include "Slices.hpp"

TEST(TestView, TestSlice) { 
    constexpr Parameter tensor(float32, {3,1,4,5}); 
    constexpr auto view = tensor[2][0][{1,3}];  
    static_assert(view.shape() == Shape(2,5));   
    static_assert(view.strides() == Strides(5, 1));
    static_assert(view.offset() == 45*dsizeof(float32));    
}  

TEST(TestView, TestSlice2) {
    constexpr Parameter tensor(int8, {6,2,3,4,4});
    constexpr auto view = tensor[{1,3}][{0,2}][1]; 
    static_assert(view.shape() == Shape(2, 2, 4, 4)); 
    static_assert(view.strides() == Strides(96, 48, 4, 1)); 
    static_assert(view.offset() == 112 * dsizeof(int8));
}
 

TEST(TestView, TestNegativeSlice) {
    constexpr Parameter tensor(int8, {6,2,3,4,4});
    constexpr auto view = tensor[{-5,-4}][{0,-1}][-2]; 
    static_assert(view.shape()[0] == 2); 
    static_assert(view.shape()[1] == 2); 
    static_assert(view.shape() == Shape(2, 2, 4, 4)); 
    static_assert(view.strides() == Strides(96, 48, 4, 1));  
    static_assert(view.offset() == 112 * dsizeof(int8));
}