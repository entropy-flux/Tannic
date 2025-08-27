#include <iostream>
#include <tannic/tensor.hpp>

using namespace tannic;

int main() {
    Tensor X = {true, true, false, true, false};
    std::cout << X << std::endl;
}
