#include <iostream>  
#include <tannic.hpp>

using namespace tannic;

// run this file with ``bash main.sh``

int main() { 
    // WARNING:
    // Explicit inialization required for now but maybe removed in the future.
    // If not properly initialized the tensors may segfault instead of throwing error. 
    // This will be fixed when resources can be infered at the end of a templated expression.

    std::cout << "Working example of the tensor library" << std::endl;

    Tensor X(float32, {2,2}); X.initialize(); // or X.initialize(Device()); for CUDA support   
    X[0, 0] = 1;
    X[0, 1] = 0;
    X[1, 0] = 2;
    X[1, 1] = 3;  
    std::cout << X << std::endl; 
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