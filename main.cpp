#include <iostream>
#include <tannic.hpp> 
#include <tannic/filter.hpp>

using namespace tannic;
 
int main() {  
    std::cout << "Working example of the tensor library" << std::endl;

    Tensor X(float16, {2,2}); // X.initialize(Device()); for CUDA support   
    X[0, range{0,-1}] = 1; 
    X[1,0] = 3;
    X[1][1] = 4;   
    
    Tensor Y(float32, {1,2}); // or X.initialize(Device()); for CUDA support   
    Y[0,0] = 4;
    Y[0,1] = 6;   
    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose());
    std::cout << Y; /* Tensor([[23.2817, 43.2817], 
                              [33.0131, 18.7881]] dtype=float32, shape=(2, 2))*/
} 


/*


(64, 16, 4, 1)
(64, 4, 16, 1)
(64, 4, 16, 2, 1)

*/