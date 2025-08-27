#include <iostream>
#include <tannic.hpp>
#include <tannic/transformations.hpp>
#include <tannic/convolutions.hpp>

using namespace tannic; 
#include <iostream>
#include <tannic.hpp>
#include <tannic/transformations.hpp>
#include <tannic/convolutions.hpp>

using namespace tannic;

int main() {
    //Implicit initialization, deduce type from initializer list.
    
    Tensor A = {1,2,3,4,5}; // int tensor
    Tensor B = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}; //float tensor
    Tensor C = {
        {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, // double
        {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
    };

    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << C << std::endl;


    // Explicit intialization, dtype and shape are defined before intilize
    // the tensors, types from initializer list will be promoted to the dtype of the
    // tensor.
    // --- 1D tensor ---
    Tensor X(float32, {3});
    X = {1, 2, 3};
    std::cout << "1D tensor X:" << std::endl;
    std::cout << X << std::endl << std::endl;

    // --- 2D tensor ---
    Tensor Y(float32, {2, 3});
    Y = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::cout << "2D tensor Y:" << std::endl;
    std::cout << Y << std::endl << std::endl;

    // --- 3D tensor ---
    Tensor Z(float32, {2, 2, 2});
    Z = {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    std::cout << "3D tensor Z:" << std::endl;
    std::cout << Z << std::endl << std::endl;

    // --- 4D tensor ---
    Tensor W(float32, {2, 2, 2, 2});
    W = {
        {
            {
                {1,  2},
                {3,  4}
            },
            {
                {5,  6},
                {7,  8}
            }
        },
        {
            {
                { 9, 10},
                {11, 12}
            },
            {
                {13, 14},
                {15, 16}
            }
        }
    };
    std::cout << "4D tensor W:" << std::endl;
    std::cout << W << std::endl << std::endl; 
} 
