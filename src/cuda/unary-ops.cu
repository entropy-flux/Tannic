#include "cuda/cuda.cuh"
#include "cuda/unary-ops.cuh"
#include <cuda_runtime.h>
#include <cassert>

template<typename S, typename D, typename Op>
__global__ void unaryOpKernel(
    const S* __restrict__ src_data,
    D* __restrict__ dst_data,
    size_t total,
    uint8_t rank,
    const size_t* __restrict__ shape,
    const size_t* __restrict__ strides_src,
    size_t offset_src,
    size_t offset_dst,
    Op op
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    size_t counters[8] = {0};  // assume max rank is 8
    size_t residual = idx;
    for (int i = rank - 1; i >= 0; --i) {
        counters[i] = residual % shape[i];
        residual /= shape[i];
    }

    size_t src_offset = 0;
    for (int i = 0; i < rank; ++i) {
        src_offset += counters[i] * strides_src[i];
    }

    dst_data[offset_dst + idx] = op(src_data[offset_src + src_offset]);
}

template<typename S, typename D, typename Op>
void cuda::unaryOp(const tensor_t* src, tensor_t* dst, Op op, cudaStream_t stream) {
    const auto rank = dst->rank;
    assert(rank <= 8 && "Rank exceeds kernel limit");

    const size_t* shape = dst->shape;
    const size_t* strides_src = src->strides;

    size_t total = 1;
    for (uint8_t i = 0; i < rank; ++i) {
        total *= shape[i];
    }

    const int threadsPerBlock = 256;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

    unaryOpKernel<S, D, Op><<<blocks, threadsPerBlock, 0, stream>>>(
        static_cast<const S*>(src->storage->address),
        static_cast<D*>(dst->storage->address),
        total,
        rank,
        shape,
        strides_src,
        src->offset,
        dst->offset,
        op
    );
    CUDA_CHECK(cudaGetLastError());

#ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
}


// TODO: Explicit template instatiation should be refactor with macros. 
// NEGATION
template void cuda::unaryOp<int8_t, int8_t, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);
template void cuda::unaryOp<int16_t, int16_t, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);
template void cuda::unaryOp<int32_t, int32_t, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);
template void cuda::unaryOp<int64_t, int64_t, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);
template void cuda::unaryOp<float, float, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::negation_op::Negation>(const tensor_t*, tensor_t*, cuda::negation_op::Negation, cudaStream_t);

// LOG
template void cuda::unaryOp<float, float, cuda::log_op::Log>(const tensor_t*, tensor_t*, cuda::log_op::Log, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::log_op::Log>(const tensor_t*, tensor_t*, cuda::log_op::Log, cudaStream_t);

// EXP
template void cuda::unaryOp<float, float, cuda::exp_op::Exp>(const tensor_t*, tensor_t*, cuda::exp_op::Exp, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::exp_op::Exp>(const tensor_t*, tensor_t*, cuda::exp_op::Exp, cudaStream_t);

// SQRT
template void cuda::unaryOp<float, float, cuda::sqrt_op::Sqrt>(const tensor_t*, tensor_t*, cuda::sqrt_op::Sqrt, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::sqrt_op::Sqrt>(const tensor_t*, tensor_t*, cuda::sqrt_op::Sqrt, cudaStream_t);

// ABS
template void cuda::unaryOp<int8_t, int8_t, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);
template void cuda::unaryOp<int16_t, int16_t, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);
template void cuda::unaryOp<int32_t, int32_t, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);
template void cuda::unaryOp<int64_t, int64_t, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);
template void cuda::unaryOp<float, float, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::abs_op::Abs>(const tensor_t*, tensor_t*, cuda::abs_op::Abs, cudaStream_t);

// SIN
template void cuda::unaryOp<float, float, cuda::sin_op::Sin>(const tensor_t*, tensor_t*, cuda::sin_op::Sin, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::sin_op::Sin>(const tensor_t*, tensor_t*, cuda::sin_op::Sin, cudaStream_t);

// COS
template void cuda::unaryOp<float, float, cuda::cos_op::Cos>(const tensor_t*, tensor_t*, cuda::cos_op::Cos, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::cos_op::Cos>(const tensor_t*, tensor_t*, cuda::cos_op::Cos, cudaStream_t);

// TAN
template void cuda::unaryOp<float, float, cuda::tan_op::Tan>(const tensor_t*, tensor_t*, cuda::tan_op::Tan, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::tan_op::Tan>(const tensor_t*, tensor_t*, cuda::tan_op::Tan, cudaStream_t);

// SINH
template void cuda::unaryOp<float, float, cuda::sinh_op::Sinh>(const tensor_t*, tensor_t*, cuda::sinh_op::Sinh, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::sinh_op::Sinh>(const tensor_t*, tensor_t*, cuda::sinh_op::Sinh, cudaStream_t);

// COSH
template void cuda::unaryOp<float, float, cuda::cosh_op::Cosh>(const tensor_t*, tensor_t*, cuda::cosh_op::Cosh, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::cosh_op::Cosh>(const tensor_t*, tensor_t*, cuda::cosh_op::Cosh, cudaStream_t);

// TANH
template void cuda::unaryOp<float, float, cuda::tanh_op::Tanh>(const tensor_t*, tensor_t*, cuda::tanh_op::Tanh, cudaStream_t);
template void cuda::unaryOp<double, double, cuda::tanh_op::Tanh>(const tensor_t*, tensor_t*, cuda::tanh_op::Tanh, cudaStream_t);