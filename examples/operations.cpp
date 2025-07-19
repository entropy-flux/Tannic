#include <iostream>
#include <tannic.hpp> 

using namespace tannic;
  
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/

int main() { 
    Tensor X(float32, {2,2});  X.initialize();
    X[0, range{0,-1}] = 1; 
    X[1,0] = 3;
    X[1,1] = 4;   
    Tensor Y(float32, {1,2});  Y.initialize();
    Y[0,0] = 4;
    Y[0,1] = 6;   
    std::cout << log(X) + Y * Y - exp(X) + matmul(X, Y.transpose());
}