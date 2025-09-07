#include <iostream>
#include <tannic.hpp> 
#include <tannic/filter.hpp>

using namespace tannic;

int main() { 
    Tensor X = -infinity(float32, {4,4});
    std::cout << triangular(X, Position::Upper) << std::endl;
}
