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

    std::cout << "Working example of the tensor library" << std::endl;

    Tensor X(float32, {2,2}); X.initialize(); // or X.initialize(Device()); for CUDA support   
    X[0, range{0,-1}] = 1; 
    X[1,0] = 3;
    X[1][1] = 4;   
    
    Tensor Y(float32, {1,2}); Y.initialize();  // or X.initialize(Device()); for CUDA support   
    Y[0,0] = 4;
    Y[0,1] = 6;   
    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose());
    std::cout << Y; /* Tensor([[23.2817, 43.2817], 
                              [33.0131, 18.7881]] dtype=float32, shape=(2, 2))*/
} 

/*
Equivalent torch code

import torch
 
X = torch.zeros((2, 2), dtype=torch.float32)
 
X[0, 0:] = 1       
X[1, 0] = 3
X[1, 1] = 4
 
Y = torch.zeros((1, 2), dtype=torch.float32)
 
Y[0, 0] = 4     
Y[0, 1] = 6       
result = torch.log(X) + Y * Y - torch.exp(X) + torch.matmul(X, Y.t())

print(result) 
tensor([[23.2817, 43.2817],
        [33.0131, 18.7881]])

*/