#include <iostream>
#include <tannic.hpp>
#include <tannic/Convolutions.hpp>

using namespace tannic;

int main() {
    // Input tensor: batch_size=1, channels=1, height=3, width=3
    Tensor input = {{
        {{1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f},
         {7.0f, 8.0f, 9.0f}}
    }};  // shape: (1,1,3,3)

    // Kernel tensor: out_channels=1, in_channels=1, height=2, width=2
    Tensor kernel = {{
        {{1.0f, 0.0f},
         {0.0f, -1.0f}}
    }};  // shape: (1,1,2,2)

    std::cout << "Input Tensor:" << std::endl;
    std::cout << input << std::endl;

    std::cout << "Kernel Tensor:" << std::endl;
    std::cout << kernel << std::endl;
 
    auto output = convolve<2>(input, kernel, {1,1}, {0,0});

    std::cout << "Output Tensor:" << std::endl;
    std::cout << output << std::endl;

    return 0;
}
