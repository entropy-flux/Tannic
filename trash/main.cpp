#include <iostream>
#include <cstdint>
#include <stdexcept>    
#include <any>
#include <cassert>
#include <array>  // You were missing this include 

enum types {
    in,
    fl,
    TYPES
};


constexpr int idx = 1;
static constexpr int arr[2] = {
    [idx] = 5,  // Still won't work in C++ – only identifiers from enums or aggregate types allowed
}; 
int main() {
    std::cout << arr[2]; 
    return 0;
}