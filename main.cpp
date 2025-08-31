#include <iostream>
#include <tannic.hpp>
#include <tannic/reductions.hpp>
#include <tannic/transformations.hpp>

using namespace tannic;

int main() {   
    Tensor A(float32, {2, 1, 3});
    A.initialize({ { {0., 1., 2.} }, { {3., 4., 5.} } });

    Tensor B(float32, {1, 4, 3});
    B.initialize({ { {0., 10., 20.}, {30., 40., 50.}, {60., 70., 80.}, {90., 100., 110.} } });

    Tensor expected(float32, {2, 4, 3});
    expected.initialize({
        {
            {  0., 11., 22. },
            { 30., 41., 52. },
            { 60., 71., 82. },
            { 90.,101.,112. }
        },
        {
            {  3., 14., 25. },
            { 33., 44., 55. },
            { 63., 74., 85. },
            { 93.,104.,115. }
        }
    });

    Tensor result = A + B;  
    std::cout << result << std::endl;

    std::cout << expected << std::endl;
} 
