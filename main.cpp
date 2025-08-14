#include <iostream>  
#include <tannic.hpp>

using namespace tannic;
 

int main() {  
    Tensor X(float32, {2,2,3}); X.initialize(); 
    X[0, 0, 0] = 1.0;
    X[0, 0, 1] = 2.0;
    X[0, 0, 2] = 3.0;

    X[0, 1, 0] = 4.0;
    X[0, 1, 1] = 5.0;
    X[0, 1, 2] = 6.0;

    X[1, 0, 0] = -1.0;
    X[1, 0, 1] = -2.0;
    X[1, 0, 2] = -3.0;

    X[1, 1, 0] = 0.5;
    X[1, 1, 1] = 1.0;
    X[1, 1, 2] = 1.5; 

    Tensor mean_sq = mean(X*X, -1); 
 
    std::cout << mean_sq.strides();
}
  