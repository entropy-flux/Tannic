#include <tannic/Tensor.hpp>

using tannic::Tensor;

int main() { 
    Tensor tensor(float32, {2,2});  tensor.initialize();
    tensor[0,0] = 1;
    tensor[0,1] = 2;
    tensor[1,0] = 3;
    tensor[1,1] = 4; 
}