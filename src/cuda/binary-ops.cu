#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "core/tensor.h"
#include "cuda/binary-ops.cuh"

template<typename S0, typename S1, typename D, typename Op>
__global__ void binaryOpKernel(tensor_t src0, tensor_t src1, tensor_t dst, size_t total, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    constexpr size_t MAX_RANK = 8;
    static_assert(MAX_RANK >= 8, "MAX_RANK");

    size_t counters[MAX_RANK] = {0};
    size_t residual = idx;
    for (int i = dst.rank - 1; i >= 0; --i) {
        counters[i] = residual % dst.shape[i];
        residual /= dst.shape[i];
    }

    size_t offset0 = 0, offset1 = 0;
    for (uint8_t i = 0; i < dst.rank; ++i) {
        offset0 += counters[i] * src0.strides[i];
        offset1 += counters[i] * src1.strides[i];
    }

    const S0* src0_data = static_cast<const S0*>(src0.address);
    const S1* src1_data = static_cast<const S1*>(src1.address);
    D* dst_data = static_cast<D*>(dst.address);

    dst_data[dst.offset + idx] = op(
        src0_data[src0.offset + offset0],
        src1_data[src1.offset + offset1]
    );
}

template<typename S0, typename S1, typename D, typename Op>
void cuda::binaryOp(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, Op op, cudaStream_t stream) {
    size_t total = 1;
    for (uint8_t i = 0; i < dst->rank; ++i) {
        total *= dst->shape[i];
    }

    const int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    binaryOpKernel<S0, S1, D, Op><<<blocks, threadsPerBlock, 0, stream>>>(
        *src0, *src1, *dst, total, op
    );
    CUDA_CHECK(cudaGetLastError());

    #ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(stream));
    #endif
}


// Addition
template void cuda::binaryOp<int8_t,  int8_t,  int8_t,  cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int8_t,  int16_t, int16_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int8_t,  int32_t, int32_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int8_t,  int64_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);

template void cuda::binaryOp<int16_t, int8_t,  int16_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int16_t, int16_t, int16_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int16_t, int32_t, int32_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int16_t, int64_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);

template void cuda::binaryOp<int32_t, int8_t,  int32_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int32_t, int16_t, int32_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int32_t, int32_t, int32_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int32_t, int64_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);

template void cuda::binaryOp<int64_t, int8_t,  int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int64_t, int16_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int64_t, int32_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int64_t, int64_t, int64_t, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);

template void cuda::binaryOp<int32_t, float,  float,  cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<float,   int32_t, float,  cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<int32_t, double, double, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<double,  int32_t, double, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);

template void cuda::binaryOp<float,  float,  float,  cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<float,  double, double, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<double, float,  double, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);
template void cuda::binaryOp<double, double, double, cuda::addition::Addition>(const tensor_t*, const tensor_t*, tensor_t*, cuda::addition::Addition, cudaStream_t);


// Subtraction
template void cuda::binaryOp<int8_t,  int8_t,  int8_t,  cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int8_t,  int16_t, int16_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int8_t,  int32_t, int32_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int8_t,  int64_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);

template void cuda::binaryOp<int16_t, int8_t,  int16_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int16_t, int16_t, int16_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int16_t, int32_t, int32_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int16_t, int64_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);

template void cuda::binaryOp<int32_t, int8_t,  int32_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int32_t, int16_t, int32_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int32_t, int32_t, int32_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int32_t, int64_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);

template void cuda::binaryOp<int64_t, int8_t,  int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int64_t, int16_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int64_t, int32_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int64_t, int64_t, int64_t, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);

template void cuda::binaryOp<int32_t, float,  float,  cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<float,   int32_t, float,  cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<int32_t, double, double, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<double,  int32_t, double, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);

template void cuda::binaryOp<float,  float,  float,  cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<float,  double, double, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<double, float,  double, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);
template void cuda::binaryOp<double, double, double, cuda::subtraction::Subtraction>(const tensor_t*, const tensor_t*, tensor_t*, cuda::subtraction::Subtraction, cudaStream_t);


// Multiplication
template void cuda::binaryOp<int8_t,  int8_t,  int8_t,  cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int8_t,  int16_t, int16_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int8_t,  int32_t, int32_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int8_t,  int64_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);

template void cuda::binaryOp<int16_t, int8_t,  int16_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int16_t, int16_t, int16_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int16_t, int32_t, int32_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int16_t, int64_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);

template void cuda::binaryOp<int32_t, int8_t,  int32_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int32_t, int16_t, int32_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int32_t, int32_t, int32_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int32_t, int64_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);

template void cuda::binaryOp<int64_t, int8_t,  int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int64_t, int16_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int64_t, int32_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int64_t, int64_t, int64_t, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);

template void cuda::binaryOp<int32_t, float,  float,  cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<float,   int32_t, float,  cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<int32_t, double, double, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<double,  int32_t, double, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);

template void cuda::binaryOp<float,  float,  float,  cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<float,  double, double, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<double, float,  double, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);
template void cuda::binaryOp<double, double, double, cuda::multiplication::Multiplication>(const tensor_t*, const tensor_t*, tensor_t*, cuda::multiplication::Multiplication, cudaStream_t);