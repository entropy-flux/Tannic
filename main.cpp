#include <iostream>
#include <tannic.hpp>
#include <tannic/transformations.hpp>

using namespace tannic;

int main() {
    Tensor X = {
        {   
            { {1.0f, 2.0f, 3.0f},
              {4.0f, 5.0f, 6.0f} },
            { {7.0f, 8.0f, 9.0f},
              {10.0f, 11.0f, 12.0f} }
        },
        {   
            { {2.0f, 3.0f, 4.0f},
              {5.0f, 6.0f, 7.0f} },
            { {8.0f, 9.0f, 10.0f},
              {11.0f, 12.0f, 13.0f} }
        }
    };
 
    Tensor W = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    }; 
    
    std::cout << matmul(X, W.transpose(-1, -2));
    return 0;
}
