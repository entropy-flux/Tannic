#include <tannic.hpp>

using namespace tannic;

int main() {
    Tensor X(float16, {2,2}); X.initialize({{1, 2}, {3, 4}});
    std::cout << X << std::endl;
}
