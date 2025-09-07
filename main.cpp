#include <iostream>
#include <tannic.hpp> 
#include <tannic/filter.hpp>

using namespace tannic;

int main() { 
    Tensor X(float32, {4,4}); X.initialize(
    {
        {1,2,3,4},
        {1,2,3,4},
        {1,2,3,4},
        {1,2,3,4},

    }, Device());
    std::cout << triangular(X, Position::Upper) << std::endl;
}
