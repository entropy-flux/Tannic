#include <iostream>

#include "Operations.hpp"
#include "Tensor.hpp" 

namespace operation {

void Negation::perform(Tensor const& operand, Tensor& result) const {
    std::cout << "Performing negation..." << std::endl;
}

void Addition::perform(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {
    std::cout << "Performing addition..." << std::endl;
}

void Subtraction::perform(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {
    std::cout << "Performing subtraction..." << std::endl;
}

void Multiplication::perform(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {
    std::cout << "Performing multiplication..." << std::endl;
}

} // namespace operation
