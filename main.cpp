#include <type_traits>
#include <initializer_list>
#include <variant>
#include <optional>
#include <iostream>
#include <vector>

#include "include/Tensor.hpp"  

int main() {
    Tensor x({1,2,3}, float32);
    x[0][0][0] = 7.f;
    x[0][0][1] = 2.f;
    x[0][0][2] = 3.f;
    x[0][1][0] = 4.f;
    x[0][1][1] = 5.f;
    x[0][1][2] = 6.f;
    std::cout << x[0][0][0].item<float>();
    Tensor y = x[0];
    std::cout << y[0][0].item<float>(); 
}