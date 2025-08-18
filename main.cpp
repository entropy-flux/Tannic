#include <iostream>
#include <type_traits>
#include <initializer_list>

struct Tensor {
    template<typename T>
    Tensor(std::initializer_list<T> const& values) {
        auto handle = [](std::initializer_list<T> const& values) {
            if constexpr (std::is_arithmetic_v<T>) {
                std::cout << "True\n";
            } else {
                std::cout << "False\n";
            }
        };
        handle(values);
    } 

    template<typename T>
    Tensor(std::initializer_list<std::initializer_list<T>> const& values) {
        auto handle = [](std::initializer_list<std::initializer_list<T>> const& values) {
            if constexpr (std::is_arithmetic_v<T>) {
                std::cout << "True\n";
            } else {
                std::cout << "False\n";
            }
        };
        handle(values);
    } 
};

int main() {
    Tensor X = {1,2,3};               // prints True
    Tensor Y = {{1,2,3},{1,2,3}};
}
