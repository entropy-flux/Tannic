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
    
    std::cout <<  X.strides() << X.shape() << std::endl;
} 


/*


(64, 16, 4, 1)
(64, 4, 16, 1)
(64, 4, 16, 2, 1)

*/