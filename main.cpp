#include <iostream>
#include <tannic.hpp> 
#include <tannic/filter.hpp>

using namespace tannic;

int main() { 
    Tensor X = -infinity(float32, {4,4});
    std::cout << exp(triangular(X, Position::Upper)) << std::endl;
}
