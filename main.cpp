#include <tannic.hpp>

using namespace tannic;

int main() {
    Tensor X(float16, {2,2}); X.initialize({{1.1, 2.0}, {3.0, 4.0}}, Device());
    Tensor Y(float32, {2,2}); Y.initialize({{4.0, 5.0}, {2.0, 3.0}}, Device());

    std::cout << Y*X + log(Y) + exp(X) << std::endl;
}