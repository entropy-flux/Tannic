#include <iostream>
#include <tannic.hpp>  

using namespace tannic; 
int main() {  
    std::cout << "Working example of the tensor library" << std::endl;

    Tensor X(float16, {2,2}); 
    X[0, range{0,-1}] = 1; 
    X[1,0] = 3;
    X[1][1] = 4;   
    
    Tensor Y(float32, {1,2});  
    Y[0,0] = 4;
    Y[0,1] = 6;   
    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose());
    std::cout <<  4 * Y;  
} 
