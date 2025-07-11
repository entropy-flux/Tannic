#include <cstddef>
#include <cstdint>
#include <cstring>
#include "cpu/unary-ops.hpp"


template<typename S, typename D, typename UnaryOp>
void cpu::unaryOp(const tensor_t* src, tensor_t* dst, UnaryOp op) {
    const S* src_data = static_cast<const S*>(src->data) + src->offset;
    D* dst_data = static_cast<D*>(dst->data) + dst->offset;

    size_t counters[8] = {0};

    for (size_t idx = 0;; ++idx) {
        size_t offset = 0;

        for (uint8_t i = 0; i < dst->rank; ++i) {
            offset += counters[i] * src->strides[i];
        }

        dst_data[idx] = op(src_data[offset]);

        bool done = false;
        for (int i = dst->rank - 1; i >= 0; --i) {
            if (++counters[i] < dst->shape[i])
                break;
            if (i == 0)
                done = true;
            counters[i] = 0;
        }

        if (done) break;
    }
}


// TODO: Explicit template instatiation should be refactor with macros. 
template void cpu::unaryOp<int8_t,  int8_t,  cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unaryOp<int16_t, int16_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unaryOp<int32_t, int32_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unaryOp<int64_t, int64_t, cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unaryOp<float,   float,   cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
template void cpu::unaryOp<double,  double,  cpu::negation::Negation>(const tensor_t*, tensor_t*, cpu::negation::Negation);
 
template void cpu::unaryOp<float,  float,  cpu::log::Log>(const tensor_t*, tensor_t*, cpu::log::Log);
template void cpu::unaryOp<double, double, cpu::log::Log>(const tensor_t*, tensor_t*, cpu::log::Log);
 
template void cpu::unaryOp<float,  float,  cpu::exp::Exp>(const tensor_t*, tensor_t*, cpu::exp::Exp);
template void cpu::unaryOp<double, double, cpu::exp::Exp>(const tensor_t*, tensor_t*, cpu::exp::Exp);
 
template void cpu::unaryOp<float,  float,  cpu::sqrt::Sqrt>(const tensor_t*, tensor_t*, cpu::sqrt::Sqrt);
template void cpu::unaryOp<double, double, cpu::sqrt::Sqrt>(const tensor_t*, tensor_t*, cpu::sqrt::Sqrt);
 
template void cpu::unaryOp<int8_t,  int8_t,  cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unaryOp<int16_t, int16_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unaryOp<int32_t, int32_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unaryOp<int64_t, int64_t, cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unaryOp<float,   float,   cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
template void cpu::unaryOp<double,  double,  cpu::abs::Abs>(const tensor_t*, tensor_t*, cpu::abs::Abs);
 
template void cpu::unaryOp<float,  float,  cpu::sin::Sin>(const tensor_t*, tensor_t*, cpu::sin::Sin);
template void cpu::unaryOp<double, double, cpu::sin::Sin>(const tensor_t*, tensor_t*, cpu::sin::Sin);
 
template void cpu::unaryOp<float,  float,  cpu::cos::Cos>(const tensor_t*, tensor_t*, cpu::cos::Cos);
template void cpu::unaryOp<double, double, cpu::cos::Cos>(const tensor_t*, tensor_t*, cpu::cos::Cos);
 
template void cpu::unaryOp<float,  float,  cpu::tan::Tan>(const tensor_t*, tensor_t*, cpu::tan::Tan);
template void cpu::unaryOp<double, double, cpu::tan::Tan>(const tensor_t*, tensor_t*, cpu::tan::Tan);
 
template void cpu::unaryOp<float,  float,  cpu::sinh::Sinh>(const tensor_t*, tensor_t*, cpu::sinh::Sinh);
template void cpu::unaryOp<double, double, cpu::sinh::Sinh>(const tensor_t*, tensor_t*, cpu::sinh::Sinh);
 
template void cpu::unaryOp<float,  float,  cpu::cosh::Cosh>(const tensor_t*, tensor_t*, cpu::cosh::Cosh);
template void cpu::unaryOp<double, double, cpu::cosh::Cosh>(const tensor_t*, tensor_t*, cpu::cosh::Cosh);
 
template void cpu::unaryOp<float,  float,  cpu::tanh::Tanh>(const tensor_t*, tensor_t*, cpu::tanh::Tanh);
template void cpu::unaryOp<double, double, cpu::tanh::Tanh>(const tensor_t*, tensor_t*, cpu::tanh::Tanh);