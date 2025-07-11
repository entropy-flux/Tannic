#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "core/tensor.h"
#include "cuda/binary-ops.cuh"

template<typename S0, typename S1, typename D, typename Op>
__global__ void binaryOpKernel(
    const S0* __restrict__ src0_data,
    const S1* __restrict__ src1_data,
    D* __restrict__ dst_data,
    size_t total,
    uint8_t rank,
    const size_t* __restrict__ shape,
    const size_t* __restrict__ strides0,
    const size_t* __restrict__ strides1,
    size_t offset0,
    size_t offset1,
    size_t dst_offset,
    Op op
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t counters[8] = {0};  
    size_t residual = idx;
    for (int i = rank - 1; i >= 0; --i) {
        counters[i] = residual % shape[i];
        residual /= shape[i];
    }

    size_t ofs0 = 0, ofs1 = 0;
    for (int i = 0; i < rank; ++i) {
        ofs0 += counters[i] * strides0[i];
        ofs1 += counters[i] * strides1[i];
    }

    dst_data[dst_offset + idx] = op(
        src0_data[offset0 + ofs0],
        src1_data[offset1 + ofs1]
    );
}

template<typename S0, typename S1, typename D, typename Op>
void cuda::binaryOp(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, Op op, cudaStream_t stream) {
    const auto rank = dst->rank;
    const size_t* shape = dst->shape;
    const size_t* strides0 = src0->strides;
    const size_t* strides1 = src1->strides;

    size_t total = 1;
    for (uint8_t i = 0; i < rank; ++i) {
        total *= shape[i];
    }

    const int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    binaryOpKernel<S0, S1, D, Op><<<blocks, threadsPerBlock, 0, stream>>>(
        static_cast<const S0*>(src0->storage.address),
        static_cast<const S1*>(src1->storage.address),
        static_cast<D*>(dst->storage.address),
        total,
        rank,
        shape,
        strides0,
        strides1,
        src0->offset,
        src1->offset,
        dst->offset,
        op
    );
    CUDA_CHECK(cudaGetLastError());

#ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}

 
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