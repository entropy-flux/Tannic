#include <iostream>
#include <tannic.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/ 

int main() {  
    Tensor X = { {-1.22474f, 0.f, 1.22474f}, {-1.22474f, 0.f, 1.22474f} }; 
    Tensor W = {{0.5f, 1.0f, 1.5f}}; 
    Tensor b = {{0.0f, 0.1f, 0.2f}}; 
    std::cout << X * W + b << std::endl;
} 
 