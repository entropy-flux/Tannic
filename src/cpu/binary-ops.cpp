#include <cstddef>
#include <cstdint>
#include <cstring>
#include "cpu/binary-ops.hpp" 

template<typename S0, typename S1, typename D, typename BinaryOp>
void cpu::binaryOp(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, BinaryOp op) {
    const S0* src0_data = static_cast<const S0*>(src0->address) + src0->offset;
    const S1* src1_data = static_cast<const S1*>(src1->address) + src1->offset;
    D* dst_data = static_cast<D*>(dst->address) + dst->offset;

    size_t counters[8] = {0};

    for (size_t idx = 0;; ++idx) {
        size_t offset0 = 0, offset1 = 0;

        for (uint8_t i = 0; i < dst->rank; ++i) {
            offset0 += counters[i] * src0->strides[i];
            offset1 += counters[i] * src1->strides[i];
        }

        dst_data[idx] = op(src0_data[offset0], src1_data[offset1]);

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



// Addition
template void cpu::binaryOp<int8_t,  int8_t,  int8_t,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int8_t,  int16_t, int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int8_t,  int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int8_t,  int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binaryOp<int16_t, int8_t,  int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int16_t, int16_t, int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int16_t, int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int16_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binaryOp<int32_t, int8_t,  int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int32_t, int16_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int32_t, int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int32_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binaryOp<int64_t, int8_t,  int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int64_t, int16_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int64_t, int32_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int64_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binaryOp<int32_t, float,  float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<float,   int32_t, float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<int32_t, double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<double,  int32_t, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binaryOp<float,  float,  float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<float,  double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<double, float,  double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binaryOp<double, double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

// Subtraction
template void cpu::binaryOp<int8_t,  int8_t,  int8_t,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int8_t,  int16_t, int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int8_t,  int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int8_t,  int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binaryOp<int16_t, int8_t,  int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int16_t, int16_t, int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int16_t, int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int16_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binaryOp<int32_t, int8_t,  int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int32_t, int16_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int32_t, int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int32_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binaryOp<int64_t, int8_t,  int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int64_t, int16_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int64_t, int32_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int64_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binaryOp<int32_t, float,  float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<float,   int32_t, float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<int32_t, double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<double,  int32_t, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binaryOp<float,  float,  float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<float,  double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<double, float,  double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binaryOp<double, double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

// Multiplication
template void cpu::binaryOp<int8_t,  int8_t,  int8_t,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int8_t,  int16_t, int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int8_t,  int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int8_t,  int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binaryOp<int16_t, int8_t,  int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int16_t, int16_t, int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int16_t, int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int16_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binaryOp<int32_t, int8_t,  int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int32_t, int16_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int32_t, int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int32_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binaryOp<int64_t, int8_t,  int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int64_t, int16_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int64_t, int32_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int64_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binaryOp<int32_t, float,  float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<float,   int32_t, float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<int32_t, double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<double,  int32_t, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binaryOp<float,  float,  float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<float,  double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<double, float,  double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binaryOp<double, double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
