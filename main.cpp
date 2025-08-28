#include <iostream>
#include <tannic.hpp>
#include <tannic/transformations.hpp>

using namespace tannic;

int main() { 
    Tensor cls = {{{100.0f, 200.0f, 300.0f}}};
 
    Tensor features(float32, {2,3,2,2}); 
    features.initialize({
        {   // batch 0
            { {1.0f, 2.0f}, {3.0f, 4.0f} },   // channel 0
            { {5.0f, 6.0f}, {7.0f, 8.0f} },   // channel 1
            { {9.0f,10.0f}, {11.0f,12.0f} }   // channel 2
        },
        {   // batch 1
            { {13.0f,14.0f}, {15.0f,16.0f} }, // channel 0
            { {17.0f,18.0f}, {19.0f,20.0f} }, // channel 1
            { {21.0f,22.0f}, {23.0f,24.0f} }  // channel 2
        }
    });

    std::cout << "Features shape: " << features.shape() << std::endl;
 
    Tensor flat = flatten(features, 2);
    std::cout << "After flatten shape: " << flat.shape() << std::endl;
    std::cout << flat << std::endl;
 
    Tensor sequence = flat.transpose(1,2);
    std::cout << "After transpose shape: " << sequence.shape() << std::endl;
    std::cout << sequence << std::endl;
 
    size_t batch_size = sequence.shape()[0];
    Tensor expanded = expand(cls, batch_size, -1, -1);
    std::cout << "Expanded CLS shape: " << expanded.shape() << std::endl;
    std::cout << expanded << std::endl;
 
    Tensor out = concatenate(expanded, sequence, /*dim=*/1);
    std::cout << "Final output shape: " << out.shape() << std::endl;
    std::cout << out << std::endl;

    return 0;
}
