#include <iostream>
#include <tannic.hpp>
#include <tannic/reductions.hpp>
#include <tannic/transformations.hpp>

using namespace tannic;

int main() {  
    Tensor X = {{2,3,4,5}, {4,5,6,7}};
    X = X.transpose();
    std::cout << X.is_contiguous() << std::endl;   
    std::cout << X.strides() << std::endl;
}
