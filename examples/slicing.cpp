#include <iostream>
#include <tannic.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/ 

int main() { 
    // WARNING:
    // Explicit inialization required for now but maybe removed in the future.
    // If not properly initialized the tensors may segfault instead of throwing error. 
    // This will be fixed when resources can be infered at the end of a templated expression.

    Tensor X(float32, {2,2,2}); X.initialize(); 
    X[0] = 1;    //arbitrary types assignment support.
    X[1] = 2.f;
    std::cout << X  << std::endl; // Tensor([[[1, 1], [1, 1]], 
                                 //          [[2, 2], [2, 2]]] dtype=float32, shape=(2, 2, 2))

    Tensor Y = X[1]; 
    std::cout << Y << std::endl; // Tensor([[2, 2], 
                                //          [2, 2]] dtype=float32, shape=(2, 2))

    std::cout << Y[0] << std::endl; // Tensor([2, 2] dtype=float32, shape=(2))
} 