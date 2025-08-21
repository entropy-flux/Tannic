#include <iostream>
#include <tannic.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/ 

int main() {  
    Tensor X = { {-1.22474f, 0.f, 1.22474f}, {-1.22474f, 0.f, 1.22474f} }; 
    Tensor W = {0.5f, 1.0f, 1.5f}; 
    Tensor b = {0.0f, 0.1f, 0.2f}; 
    std::cout << X * W + b << std::endl;

    /*
    
Tensor([[-0.61237, 0.1, 2.03711], 
        [-0.61237, 0.1, 2.03711]] dtype=float32, shape=(2, 3))
    
    */
} 
 
/*
Weight: Tensor([[0.5, 1, 1.5]] dtype=float32, shape=(1, 3))
Bias: Tensor([[0, 0.1, 0.2]] dtype=float32, shape=(1, 3))
Normalized:Tensor([[-1.22474, 0, 1.22474], 
        [-1.22474, 0, 1.22474]] dtype=float32, shape=(2, 3))
 
*/