#include "Tensor.hpp"
#include "Algebra/Operations.hpp"
#include "Algebra/Transformations.hpp"
#include "cpu/kernels.h"


 
void Negation::forward(Tensor const& operand, Tensor& result) const {  
    cpu::negation::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    ); 
}

void Addition::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {  
    
    if(!(operand.shape() == cooperand.shape())) {
        throw std::runtime_error("Broadcast supported but not implemented for this kernel.");
    }

    cpu::addition::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), cooperand.address(), result.address(), operand.shape().size()
    ); 
}

void Subtraction::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {  
    
    if(!(operand.shape() == cooperand.shape())) {
        throw std::runtime_error("Broadcast supported but not implemented for this kernel.");
    }

    cpu::subtraction::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), cooperand.address(), result.address(), operand.shape().size()
    ); 
}


void Multiplication::forward(Tensor const& operand, Tensor const& cooperand, Tensor& result) const {  

    if(!(operand.shape() == cooperand.shape())) {
        throw std::runtime_error("Broadcast supported but not implemented for this kernel.");
    }

    cpu::multiplication::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), cooperand.address(), result.address(), operand.shape().size()
    ); 
}

void Matmul::forward(Tensor const& multiplicand, Tensor const& multiplier, Tensor& result) const {   
    void const* A = multiplicand.address();
    void const* B = multiplier.address();
    void* C = result.address();
    auto M = multiplicand.shape().size() / multiplicand.shape().back();
    auto K = multiplicand.shape().back();
    auto N = multiplier.shape().back(); 
    
    switch((multiplicand.is_transposed() ? 2 : 0) + (multiplier.is_transposed() ? 1 : 0)) {
        case 0: 
            cpu::matmul::kernels[multiplicand.dtype()][multiplier.dtype()](A, B, C, M, N, K, false);
            break;

        case 1: 
            cpu::matmul::kernels[multiplicand.dtype()][multiplier.dtype()](A, B, C, M, N, K, true);
            break;

        case 2:  
            throw std::runtime_error("Not implemnented yet.");
            break; 

        case 3:  
            throw std::runtime_error("Not implemnented yet.");
            break; 
    }
}