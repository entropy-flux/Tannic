#include <iostream>
#include <tannic/Tensor.hpp>
#include <tannic/Transformations.hpp>

using namespace tannic;

int main() { 
    Tensor X(float32, {2,2});  X.initialize();
    X[0][{0,2}] = 1; 
    X[1,0] = 3;
    X[1,1] = 4;   
    Tensor Y(float32, {1,2});  Y.initialize();
    Y[0,0] = 4;
    Y[0,1] = 6;   
    std::cout << X + Y * Y - X + matmul(X, Y.transpose());
}