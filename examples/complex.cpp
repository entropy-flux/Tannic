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

    Tensor X(float32, {2,2});  // or X.initialize(Device()); for CUDA support 
    X[0, 0] = 1;
    X[0, 1] = 6;
    X[1, 0] = 2;
    X[1, 1] = 3;     
    X = complexify(X);
    
    Tensor Y(float32, {2,2});  
    Y[0, 0] = 2;
    Y[0, 1] = 1;
    Y[1, 0] = 1.5;
    Y[1, 1] = 3.14;  
    Y = complexify(Y);

    std::cout << X << std::endl;
    std::cout << Y << std::endl;
    std::cout << X * Y << std::endl;
}

/*import torch

# Create magnitude tensor X
X = torch.tensor([[1.0, 6.0],
                  [2.0, 3.0]])

# Create phase tensor Y (in radians)
Y = torch.tensor([[2.0, 1.0],
                  [1.5, 3.14]])

# Create complex tensor Z from polar coordinates
Z = torch.polar(X, Y)
 

*/