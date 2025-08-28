#include <iostream>
#include <tannic/tensor.hpp>

using namespace tannic;

int main() {  

    Tensor X = {true, true, false, true, false};
    std::cout << "X = " << X << std::endl;

    
    Tensor Y = {
        {true, false, true},
        {false, true, false}
    };
    
    std::cout << "Y = " << Y << std::endl;

    
    Tensor Z = {
        {
            {true, false},
            {false, true}
        },
        {
            {false, true},
            {true, false}
        }
    };
    std::cout << "Z = " << Z << std::endl;
 
    Tensor W = {
        {
            {
                {true, false},
                {false, true}
            },
            {
                {false, true},
                {true, false}
            }
        },
        {
            {
                {true, true},
                {false, false}
            },
            {
                {false, false},
                {true, true}
            }
        }
    };
    std::cout << "W = " << W << std::endl; 

 


    Tensor A(boolean, {5});
    A.initialize({true, true, false, true, false});
    std::cout << "A = " << A << std::endl;
 
    Tensor B(boolean, {2, 3});
    B.initialize({
        {true, false, true},
        {false, true, false}
    });
    std::cout << "B = " << B << std::endl;
 
    Tensor C(boolean, {2, 2, 2});
    C.initialize({
        {
            {true, false},
            {false, true}
        },
        {
            {false, true},
            {true, false}
        }
    });
    std::cout << "C = " << C << std::endl;
 
    Tensor D(boolean, {2, 2, 2, 2});
    D.initialize({
        {
            {
                {true, false},
                {false, true}
            },
            {
                {false, true},
                {true, false}
            }
        },
        {
            {
                {true, true},
                {false, false}
            },
            {
                {false, false},
                {true, true}
            }
        }
    });
    std::cout << "D = " << D << std::endl; 

    return 0;
}

