#include <iostream>
#include <tannic.hpp> 
#include <tannic/filter.hpp>

using namespace tannic;
 
int main() {  
    Tensor X = {1, 2, 3, 4, 5};
    Tensor Y = X;         // Cheap copy,  
    Tensor Z = Y[{1, 3}]; // Cheap view
    Z[{0, -1}] = 5;
    std::cout << X << std::endl;
} 


/*


(64, 16, 4, 1)
(64, 4, 16, 1)
(64, 4, 16, 2, 1)

*/