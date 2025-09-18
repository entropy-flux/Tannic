#include <iostream>
#include <tannic.hpp> 
#include <tannic/functions.hpp>
#include <tannic/transformations.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/ 

int main() {  
    std::cout << "Working example of the tensor library" << std::endl;

    Tensor X(float16, {2,2}); // X.initialize(Device()); for CUDA support   
    X[0, range{0,-1}] = 1; 
    X[1,0] = 3;
    X[1][1] = 4;   
    
    Tensor Y = X.to(Device()); 
    Tensor Z = Y.to(Host());
    std::cout << Z * X << std::endl;
} 
/*


(64, 16, 4, 1)
(64, 4, 16, 1)
(64, 4, 16, 2, 1)

*/