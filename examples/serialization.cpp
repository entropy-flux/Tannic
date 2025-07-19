#include <tannic.hpp> 
#include <tannic/Serialization.hpp>

using namespace tannic;
 
/*
Copy and paste this file into main.cpp and then run ``bash main.sh``
*/

int main() {
    Tensor tensor(float32, {2,5,3,4}); tensor.initialize();  
    tensor[{0,-1}] = 5;
    std::cout << tensor;
    Blob serialized = serialize(tensor);
    Tensor deserialized = deserialize(serialized);  
    std::cout << deserialized;   
}