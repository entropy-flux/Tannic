#include <iostream>
#include <tannic.hpp>
#include <tannic/comparisons.hpp>

using namespace tannic;

int main() {
    Tensor A = {1, 2, 3, 4, 5};
    Tensor B = {5, 4, 3, 2, 1};

    std::cout << "A: " << A << std::endl;
    std::cout << "B: " << B << std::endl;

    Tensor eq = (A == B);
    Tensor ne = (A != B);
    Tensor gt = (A >  B);
    Tensor ge = (A >= B);
    Tensor lt = (A <  B);
    Tensor le = (A <= B);

    std::cout << "A == B: " << eq << std::endl;
    std::cout << "A != B: " << ne << std::endl;
    std::cout << "A >  B: " << gt << std::endl;
    std::cout << "A >= B: " << ge << std::endl;
    std::cout << "A <  B: " << lt << std::endl;
    std::cout << "A <= B: " << le << std::endl;

    Tensor X = {{1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}};

    Tensor Y = {{1.0f, 2.5f, 2.5f},
                {5.0f, 5.0f, 7.0f}};

    Tensor cmp = (X < Y);
    std::cout << "X < Y: " << cmp << std::endl;

    Tensor Z(float32, {2,2,2});
    Z = {{{1, 2}, {3, 4}},
         {{5, 6}, {7, 8}}};

    Tensor W(float32, {2,2,2});
    W = {{{0, 2}, {3, 5}},
         {{5, 7}, {8, 8}}};

    Tensor mask = (Z != W);
    std::cout << "Z != W: " << mask << std::endl;
}