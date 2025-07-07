#include <cstddef>
#include <cstdint>
#include <cstring>
#include "cpu/unaryops.hpp"

template<typename TA, typename TB, typename Op>
void cpu::unary_op(const tensor_t* A, tensor_t* B, Op op) {
    const TA* A_data = static_cast<const TA*>(A->address) + A->offset;
    TB* B_data = static_cast<TB*>(B->address) + B->offset;

    size_t counters[8] = {0};

    for (size_t idx = 0;; ++idx) {
        size_t offset_A = 0;

        for (uint8_t i = 0; i < B->rank; ++i) {
            offset_A += counters[i] * A->strides[i];
        }

        B_data[idx] = op(A_data[offset_A]);

        bool done = false;
        for (int i = B->rank - 1; i >= 0; --i) {
            if (++counters[i] < B->shape[i])
                break;
            if (i == 0) {
                done = true;
            }
            counters[i] = 0;
        }

        if (done) break;
    }
}


template void cpu::unary_op<int8_t,  int8_t,  cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unary_op<int16_t, int16_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unary_op<int32_t, int32_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unary_op<int64_t, int64_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unary_op<float,   float,   cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unary_op<double,  double,  cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
 
template void cpu::unary_op<float,  float,  cpu::log::Log>(const tensor_t*, tensor_t*, cpu::log::Log);
template void cpu::unary_op<double, double, cpu::log::Log>(const tensor_t*, tensor_t*, cpu::log::Log);
 
template void cpu::unary_op<float,  float,  cpu::exp::Exp>(const tensor_t*, tensor_t*, cpu::exp::Exp);
template void cpu::unary_op<double, double, cpu::exp::Exp>(const tensor_t*, tensor_t*, cpu::exp::Exp);
 
template void cpu::unary_op<float,  float,  cpu::sqrt::Sqrt>(const tensor_t*, tensor_t*, cpu::sqrt::Sqrt);
template void cpu::unary_op<double, double, cpu::sqrt::Sqrt>(const tensor_t*, tensor_t*, cpu::sqrt::Sqrt);
 
template void cpu::unary_op<int8_t,  int8_t,  cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unary_op<int16_t, int16_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unary_op<int32_t, int32_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unary_op<int64_t, int64_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unary_op<float,   float,   cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unary_op<double,  double,  cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
 
template void cpu::unary_op<float,  float,  cpu::sin::Sin>(const tensor_t*, tensor_t*, cpu::sin::Sin);
template void cpu::unary_op<double, double, cpu::sin::Sin>(const tensor_t*, tensor_t*, cpu::sin::Sin);
 
template void cpu::unary_op<float,  float,  cpu::cos::Cos>(const tensor_t*, tensor_t*, cpu::cos::Cos);
template void cpu::unary_op<double, double, cpu::cos::Cos>(const tensor_t*, tensor_t*, cpu::cos::Cos);
 
template void cpu::unary_op<float,  float,  cpu::tan::Tan>(const tensor_t*, tensor_t*, cpu::tan::Tan);
template void cpu::unary_op<double, double, cpu::tan::Tan>(const tensor_t*, tensor_t*, cpu::tan::Tan);
 
template void cpu::unary_op<float,  float,  cpu::sinh::Sinh>(const tensor_t*, tensor_t*, cpu::sinh::Sinh);
template void cpu::unary_op<double, double, cpu::sinh::Sinh>(const tensor_t*, tensor_t*, cpu::sinh::Sinh);
 
template void cpu::unary_op<float,  float,  cpu::cosh::Cosh>(const tensor_t*, tensor_t*, cpu::cosh::Cosh);
template void cpu::unary_op<double, double, cpu::cosh::Cosh>(const tensor_t*, tensor_t*, cpu::cosh::Cosh);
 
template void cpu::unary_op<float,  float,  cpu::tanh::Tanh>(const tensor_t*, tensor_t*, cpu::tanh::Tanh);
template void cpu::unary_op<double, double, cpu::tanh::Tanh>(const tensor_t*, tensor_t*, cpu::tanh::Tanh);