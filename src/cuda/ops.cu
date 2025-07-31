#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/ops.cuh"
#include "cuda/streams.cuh"

template<typename S, typename D, class Op>
__global__ void scalarUnaryOpKernel(const S* src, D* dst) {
    Op op;
    *dst = op(*src);
}

template<typename S, typename D, class Op>
__global__ void batchedUnaryOpKernel(
    const S* src, const uint32_t* src_sz, const int64_t* src_ne,           
    D* dst, const uint32_t* dst_sz, const int64_t* dst_ne, 
    uint8_t rank, size_t ne
) {
    Op op{};

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) { 
        size_t offs = 0;
        size_t remaining = idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % dst_sz[dim];
            remaining /= dst_sz[dim];
 
            size_t src_idx = (src_sz[dim] == 1) ? 0 : dim_idx;
            offs += src_idx * src_ne[dim];
        }

        dst[idx] = op(src[offs]);
    }
}

template<typename S, typename D, class Op>
void launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, cudaStream_t stream = 0) { 
    if (src->rank == 0) {
        scalarUnaryOpKernel<S, D, Op><<<1, 1, 0,stream>>>(
            (const S*)(src->address),
            (D*)(dst->address)
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedUnaryOpKernel<S, D, Op><<<gridSize, blockSize, 0, stream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
}

template<typename S0, typename S1, typename D, class Op>
__global__ void scalarBinaryOpKernel(const S0* src0, const S1* src1, D* dst) {
    Op op;
    *dst = op(*src0, *src1);
}

template<typename S0, typename S1, typename D, class Op>
__global__ void batchedBinaryOpKernel(
    const S0* src0_ptr, const uint32_t* src0_sz, const int64_t* src0_ne,
    const S1* src1_ptr, const uint32_t* src1_sz, const int64_t* src1_ne,
    D* dst_ptr, const uint32_t* dst_sz, const int64_t* dst_ne,
    uint8_t rank, size_t ne
) {
    Op op{};  
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne;  idx += blockDim.x * gridDim.x) {
        size_t cnt[8] = {0};
        size_t remaining = idx;
         
        for (uint8_t i = rank - 1; i > 0; --i) {
            cnt[i] = remaining % dst_sz[i];
            remaining /= dst_sz[i];
        }
        cnt[0] = remaining;
         
        size_t offs0 = 0, offs1 = 0;
        for (uint8_t i = 0; i < rank; ++i) {
            size_t idx0 = (src0_sz[i] == 1) ? 0 : cnt[i];
            size_t idx1 = (src1_sz[i] == 1) ? 0 : cnt[i];
            
            offs0 += idx0 * src0_ne[i];
            offs1 += idx1 * src1_ne[i];
        }
         
        dst_ptr[idx] = op(src0_ptr[offs0], src1_ptr[offs1]);
    }
}
  
 
template<typename S, typename D, class Op>
void launchUnaryOpKernel(const tensor_t* src, tensor_t* dst) {
    if (src->rank == 0) {
        scalarUnaryOpKernel<S, D, Op>(
            (const S*)(src->address), 
            (D*)(dst->address)
        ); 
    } 
    
    else {    
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape[dim];
        }

        batchedUnaryOpKernel<S, D, Op>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
    return;
}        

void launchDefaultBinaryOpKernel(const tensor_t* src0, const tensor_t* sr1, tensor_t* dst, cudaStream_t) {
    exit(EXIT_FAILURE);
};  
  