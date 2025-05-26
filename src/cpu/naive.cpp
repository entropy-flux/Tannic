#include "Operations.hpp"
#include "Tensor.hpp"
#include "View.hpp"
#include <iostream>

namespace operation {

void Negation::perform(View const& operand, Tensor& result) const {
    std::cout << "Performing negation..." << std::endl;
}

void Addition::perform(View const& operand, View const& cooperand, Tensor& result) const {
    std::cout << "Performing addition..." << std::endl;
}

void Subtraction::perform(View const& operand, View const& cooperand, Tensor& result) const {
    std::cout << "Performing subtraction..." << std::endl;
}

void Multiplication::perform(View const& operand, View const& cooperand, Tensor& result) const {
    std::cout << "Performing multiplication..." << std::endl;
}

} // namespace operation
