#include <iostream>
#include <tannic.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/ 

int main() {  
    Tensor X = {1,2,3,4,5};
    Tensor Y = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Tensor Z = {
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}
    };

    std::cout << X << std::endl;
    std::cout << Y << std::endl;
    std::cout << Z << std::endl;
} 
