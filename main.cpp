#include <iostream>
#include <tannic.hpp>  

using namespace tannic;

#include <iostream>
#include <limits>
#include <tannic.hpp>

using namespace tannic;

int main() {
    Tensor X(float32, {2,2});
 
    X[0,0] = std::numeric_limits<float>::infinity();    // +inf
    X[0,1] = -std::numeric_limits<float>::infinity();   // -inf
    X[1,0] = 3.0f;
    X[1,1] = 4.0f;

    std::cout << "Explicit inf assignment:\n" << X;

    return 0;
}
