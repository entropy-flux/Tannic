#include <cuda_runtime.h>
#include <cassert> 
#include <vector>

#include "cuda/cuda.cuh"
#include "cuda/matmul-op.cuh"

template<typename S, typename D, typename TC>
__global__ void gemmKernel(
    const S* __restrict__ A,
    const S* __restrict__ B,
    TC* __restrict__ C,
    size_t M, size_t N, size_t K,
    size_t lda0, size_t lda1,
    size_t ldb0, size_t ldb1,
    size_t ldc0, size_t ldc1,
    bool transA, bool transB
) {
    size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    TC sum = TC(0);

    if (transA && transB) { 
        for (size_t k = 0; k < K; ++k) {
            size_t a_idx = k * lda0 + m * lda1;   
            size_t b_idx = k * ldb1 + n * ldb0;   
            sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
        }

    } else if (transA && !transB) {
        for (size_t k = 0; k < K; ++k) {
            size_t a_idx = k * lda0 + m * lda1;
            size_t b_idx = k * ldb0 + n * ldb1;
            sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
        }
    } else if (!transA && transB) {
        for (size_t k = 0; k < K; ++k) {
            size_t a_idx = m * lda0 + k * lda1;
            size_t b_idx = n * ldb0 + k * ldb1;
            sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
        }
    } else {  
        for (size_t k = 0; k < K; ++k) {
            size_t a_idx = m * lda0 + k * lda1;
            size_t b_idx = k * ldb0 + n * ldb1;
            sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
        }
    }

    C[m * ldc0 + n * ldc1] = sum;
}

template<typename S, typename D, typename TC>
void cuda::matmulOp(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool transA, bool transB, cudaStream_t stream) {
    const size_t batch_rank = dst->rank > 2 ? dst->rank - 2 : 0;
    size_t batch_size = 1;
    for (size_t i = 0; i < batch_rank; ++i)
        batch_size *= dst->shape[i];

    size_t M = dst->shape[dst->rank - 2];
    size_t N = dst->shape[dst->rank - 1];
    size_t K = transA ? src0->shape[src0->rank-2] : src0->shape[src0->rank-1];

    dim3 block(16, 16);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);

    auto unravel = [&](size_t b, size_t* idxs){
        for (int i = batch_rank-1, r=b; i >= 0; --i) {
            idxs[i] = r % dst->shape[i];
            r /= dst->shape[i];
        }
    };

    std::vector<size_t> dst_idx(batch_rank), s0_idx(batch_rank), s1_idx(batch_rank);
    
    for (size_t b = 0; b < batch_size; ++b) {
        if (batch_rank)
            unravel(b, dst_idx.data());

        for (size_t i = 0; i < batch_rank; ++i) {
            s0_idx[i] = (src0->rank > 2 && src0->shape[i] == 1) ? 0 : dst_idx[i];
            s1_idx[i] = (src1->rank > 2 && src1->shape[i] == 1) ? 0 : dst_idx[i];
        }

        size_t off0 = 0, off1 = 0, offD = 0;
        for (size_t i = 0; i < batch_rank; ++i) {
            off0 += s0_idx[i] * src0->strides[i];
            off1 += s1_idx[i] * src1->strides[i];
            offD += dst_idx[i] * dst->strides[i];
        }

        gemmKernel<S,D,TC><<<grid, block, 0, stream>>>(
            static_cast<const S*>(src0->storage->address) + src0->offset + off0,
            static_cast<const S*>(src1->storage->address) + src1->offset + off1,
            reinterpret_cast<TC*>(dst->storage->address) + dst->offset + offD,
            M, N, K,
            src0->strides[src0->rank-2],
            src0->strides[src0->rank-1],
            src1->strides[src1->rank-2],
            src1->strides[src1->rank-1],
            dst->strides[dst->rank-2],
            dst->strides[dst->rank-1],
            transA, transB
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
 

// TODO: Explicit template instatiation should be refactor with macros. 
template void cuda::matmulOp<int8_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int8_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int8_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int8_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

template void cuda::matmulOp<int16_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int16_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int16_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int16_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

template void cuda::matmulOp<int32_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int32_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int32_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int32_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

template void cuda::matmulOp<int64_t, int8_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int64_t, int16_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int64_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int64_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

template void cuda::matmulOp<float, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<float, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<double, float, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<double, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);

template void cuda::matmulOp<int32_t, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<float, int32_t, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<int32_t, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);
template void cuda::matmulOp<double, int32_t, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool, cudaStream_t);