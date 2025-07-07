#include <cstddef>
#include <cstdint>
#include <cstring>
#include "cpu/binaryops.hpp" 

template<typename TA, typename TB, typename TC, typename BinaryOp>
void cpu::binary_op(const tensor_t* A, const tensor_t* B, tensor_t* C, BinaryOp op) {
    const TA* A_data = static_cast<const TA*>(A->address) + A->offset;
    const TB* B_data = static_cast<const TB*>(B->address) + B->offset;
    TC* C_data = static_cast<TC*>(C->address) + C->offset;

    size_t counters[8] = {0};

    for (size_t idx = 0;; ++idx) {
        size_t offset_A = 0, offset_B = 0;

        for (uint8_t i = 0; i < C->rank; ++i) {
            offset_A += counters[i] * A->strides[i];
            offset_B += counters[i] * B->strides[i];
        }

        C_data[idx] = op(A_data[offset_A], B_data[offset_B]);

        bool done = false;
        for (int i = C->rank - 1; i >= 0; --i) {
            if (++counters[i] < C->shape[i])
                break;
            if (i == 0) {
                done = true;
            }
            counters[i] = 0;
        }

        if (done) break;
    }
}  

// Addition
template void cpu::binary_op<int8_t,  int8_t,  int8_t,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int8_t,  int16_t, int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int8_t,  int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int8_t,  int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binary_op<int16_t, int8_t,  int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int16_t, int16_t, int16_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int16_t, int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int16_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binary_op<int32_t, int8_t,  int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int32_t, int16_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int32_t, int32_t, int32_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int32_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binary_op<int64_t, int8_t,  int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int64_t, int16_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int64_t, int32_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int64_t, int64_t, int64_t, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binary_op<int32_t, float,  float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<float,   int32_t, float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<int32_t, double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<double,  int32_t, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

template void cpu::binary_op<float,  float,  float,  cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<float,  double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<double, float,  double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);
template void cpu::binary_op<double, double, double, cpu::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cpu::addition::Addition);

// Subtraction
template void cpu::binary_op<int8_t,  int8_t,  int8_t,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int8_t,  int16_t, int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int8_t,  int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int8_t,  int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binary_op<int16_t, int8_t,  int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int16_t, int16_t, int16_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int16_t, int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int16_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binary_op<int32_t, int8_t,  int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int32_t, int16_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int32_t, int32_t, int32_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int32_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binary_op<int64_t, int8_t,  int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int64_t, int16_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int64_t, int32_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int64_t, int64_t, int64_t, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binary_op<int32_t, float,  float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<float,   int32_t, float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<int32_t, double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<double,  int32_t, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

template void cpu::binary_op<float,  float,  float,  cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<float,  double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<double, float,  double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);
template void cpu::binary_op<double, double, double, cpu::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cpu::subtraction::Subtraction);

// Multiplication
template void cpu::binary_op<int8_t,  int8_t,  int8_t,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int8_t,  int16_t, int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int8_t,  int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int8_t,  int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binary_op<int16_t, int8_t,  int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int16_t, int16_t, int16_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int16_t, int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int16_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binary_op<int32_t, int8_t,  int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int32_t, int16_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int32_t, int32_t, int32_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int32_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binary_op<int64_t, int8_t,  int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int64_t, int16_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int64_t, int32_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int64_t, int64_t, int64_t, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binary_op<int32_t, float,  float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<float,   int32_t, float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<int32_t, double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<double,  int32_t, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);

template void cpu::binary_op<float,  float,  float,  cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<float,  double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<double, float,  double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);
template void cpu::binary_op<double, double, double, cpu::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cpu::multiplication::Multiplication);