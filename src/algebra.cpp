#include <memory>
#include <cstring>

#include "Tensor.hpp"
#include "Operations.hpp"
#include "Transformations.hpp"
#include "Functions.hpp"
#include "Modules.hpp"

#include "kernels/cpu/fun.h"
#include "kernels/cpu/ops.h"
#include "kernels/cpu/matmul.h"

#ifdef OPENBLAS
#include "kernels/openblas/copy.h"
#include "kernels/openblas/axpy.h"
#include "kernels/openblas/gemm.h"
#include "kernels/openblas/scal.h"
#endif

namespace symbol {
 
void Negation::forward(const Tensor& operand, Tensor& result) const { 
#ifdef OPENBLAS
    openblas::copy::kernels[operand.dtype()](
        operand.size(),
        operand.address(), 1,
        result.address(), 1
    );
 
    double alpha = -1.0;
    openblas::scal::kernels[operand.dtype()](
        result.size(),
        alpha,
        result.address(), 1
    );
#else
    cpu::negation::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    ); 
#endif
}


void Subtraction::forward(const Tensor& operand, const Tensor& cooperand, Tensor& result) const { 
    assert(operand.size() == cooperand.size() && cooperand.size() == result.size() && "Broadcasting not supported for openblas kernel.");

#ifdef OPENBLAS 
    assert(operand.dtype() == cooperand.dtype() && cooperand.dtype() == result.dtype() && "Mixed types not supported in openblas.");
    openblas::copy::kernels[operand.dtype()](
        cooperand.size(),
        operand.address(), 1,
        result.address(), 1
    );
 
    double alpha = -1.0;
    openblas::axpy::kernels[cooperand.dtype()](
        cooperand.size(),
        alpha,
        cooperand.address(), 1,
        result.address(), 1
    );
#else
    cpu::subtraction::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), 
        cooperand.address(), 
        result.address(), 
        operand.shape().size()
    ); 
#endif  
}


void Addition::forward(const Tensor& operand, const Tensor& cooperand, Tensor& result) const {
    assert(operand.size() == cooperand.size() && cooperand.size() == result.size() && "Broadcasting not supported for this kernel.");

    
#ifdef OPENBLAS
    assert(operand.dtype() == cooperand.dtype() && cooperand.dtype() == result.dtype() && "Mixed types not supported in openblas.");
    openblas::copy::kernels[cooperand.dtype()](
        cooperand.size(),
        cooperand.address(), 1,
        result.address(), 1
    );

    double alpha = 1.0;
    openblas::axpy::kernels[augend.dtype()](
        cooperand.size(),
        alpha,
        operand.address(), 1,
        result.address(), 1
    );
#else
    cpu::addition::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), 
        cooperand.address(), 
        result.address(), 
        operand.shape().size()
    ); 
#endif
}


void Multiplication::forward(const Tensor& operand, const Tensor& cooperand, Tensor& result) const { 
    assert(operand.size() == cooperand.size() && cooperand.size() == result.size() && "Broadcasting not supported for this kernel.");
    cpu::multiplication::kernels[operand.dtype()][cooperand.dtype()](
        operand.address(), 
        cooperand.address(), 
        result.address(), 
        operand.shape().size()
    );  
} 



void Linear::forward(Tensor const& multiplicand, Tensor const& multiplier, Tensor& result) const {   
    assert(multiplicand.dtype() == multiplier.dtype() && "Mixed types not supported for openblas backend.");  
    auto M = multiplicand.shape().size() / multiplicand.shape().back();
    auto K = multiplicand.shape().back();
    auto N = multiplier.shape().back();  
#ifdef OPENBLAS
    openblas::gemm::kernels[multiplicand.dtype()][multiplier.dtype()](
        openblas::Order::RowMajor,
        multiplicand.is_transposed() ? openblas::Transposed::True : openblas::Transposed::False,
        multiplier.is_transposed() ?  openblas::Transposed::True : openblas::Transposed::False,
        M, N, K,
        1.,
        multiplicand.address(), multiplicand.is_transposed() ? M : K,
        multiplier.address(), multiplier.is_transposed() ? K : N,
        0.,
        result.address(), N
    );
#else
    cpu::linear::kernels[multiplicand.dtype()][multiplier.dtype()](
        true, multiplicand.is_transposed(), multiplier.is_transposed(),
        M, N, K,
        multiplicand.address(), multiplicand.is_transposed() ? M : K,
        multiplier.address(), multiplier.is_transposed() ? K : N, 
        result.address(), N
    );
#endif
}
 
 

void Log::forward(Tensor const& operand, Tensor& result) const { 
    cpu::log::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Exp::forward(Tensor const& operand, Tensor& result) const { 
    cpu::exp::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Sqrt::forward(Tensor const& operand, Tensor& result) const { 
    cpu::sqrt::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Abs::forward(Tensor const& operand, Tensor& result) const { 
    cpu::abs::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Sin::forward(Tensor const& operand, Tensor& result) const { 
    cpu::sin::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Sinh::forward(Tensor const& operand, Tensor& result) const { 
    cpu::sinh::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Cos::forward(Tensor const& operand, Tensor& result) const { 
    cpu::cos::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Cosh::forward(Tensor const& operand, Tensor& result) const { 
    cpu::cosh::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Tan::forward(Tensor const& operand, Tensor& result) const {  
    cpu::tan::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}
 
void Tanh::forward(Tensor const& operand, Tensor& result) const { 
    cpu::tanh::kernels[operand.dtype()](
        operand.address(), result.address(), operand.shape().size()
    );
}

} // symbol

void Embedding::forward(Tensor& result, Tensor const& lookup) const { 
    for(auto dimension = 0; dimension < lookup.size(); dimension++) { 
        std::ptrdiff_t offset = dimension*dsizeof(lookup.dtype());
        std::memcpy(
            result.address() + offset, 
            address() + offset, 
            shape_.back() * dsizeof(dtype_)
        );
    }
}