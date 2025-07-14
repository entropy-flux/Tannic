#include "mcls.hpp"
#include <iostream>

template <typename T>
void MyClass::templateFunction(T value) {
    std::cout << "Template function called with: " << value << std::endl;
}

void MyClass::regularFunction() {
    std::cout << "Regular function called" << std::endl;
}

// Explicit template instantiations
template void MyClass::templateFunction<int>(int);
template void MyClass::templateFunction<double>(double);