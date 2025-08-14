#include <cstdint>
#include <array> 
#include <cuda_runtime.h> 
#include "cuda/argred.cuh"

namespace {

template<typename S, typename D, typename A, class Op>
__global__ void argReduceKernel(
    const S* __restrict__ src_ptr, const shape_t src_shape, const strides_t src_strides,
    D* __restrict__ dst_ptr, const shape_t /*dst_shape*/, const strides_t /*dst_strides*/,
    uint8_t rank, uint8_t dim, size_t ne
) {
    Op op;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
 
        size_t cnt[8] = {0};
        size_t remaining = idx;
        for (int d = rank - 1; d >= 0; --d) {
            if (d == dim) continue;
            const size_t sz = src_shape.sizes[d];
            cnt[d] = (sz ? (remaining % sz) : 0);
            remaining = (sz ? (remaining / sz) : 0);
        }
 
        A accum = A(0);
        const size_t reduceN = src_shape.sizes[dim];
        for (size_t i = 0; i < reduceN; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                const size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d]; // assumes strides in elements
            }
            accum = op(accum, static_cast<A>(src_ptr[offset]));
        }
 
        dst_ptr[idx] = static_cast<D>(op.finalize(accum, reduceN));
    }
}

struct Sum {
    template<typename A, typename S>
    __device__ A operator()(A accum, S val) const { return accum + static_cast<A>(val); }
    template<typename A>
    __device__ A finalize(A accum, size_t) const { return accum; }
};

struct Mean {
    template<typename A, typename S>
    __device__ A operator()(A accum, S val) const { return accum + static_cast<A>(val); }
    template<typename A>
    __device__ A finalize(A accum, size_t count) const {
        return accum / static_cast<A>(count);
    }
};

template<typename S, typename D, typename A, class Op>
status launchArgReduce(const tensor_t* src, tensor_t* dst, uint8_t dim, stream_t stream) {
    if (src->rank == 0 || dim >= src->rank) return SUCCESS; // or an error code if you prefer
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);

    size_t ne = 1;
    for (int i = 0; i < src->rank; ++i) if (i != dim) ne *= src->shape.sizes[i];

    const int blockSize = 256;
    const int gridSize  = static_cast<int>((ne + blockSize - 1) / blockSize);

    argReduceKernel<S, D, A, Op><<<gridSize, blockSize, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<D*>(dst->address), dst->shape, dst->strides,
        src->rank, dim, ne
    );
 
    return (cudaGetLastError() == cudaSuccess) ? SUCCESS : ERROR;
}

} namespace cuda {

status argsum(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgReduce<int8_t,  int8_t,  int64_t, Sum>(src, dst, dim, stream);
        case int16:   return launchArgReduce<int16_t, int16_t, int64_t, Sum>(src, dst, dim, stream);
        case int32:   return launchArgReduce<int32_t, int32_t, int64_t, Sum>(src, dst, dim, stream);
        case int64:   return launchArgReduce<int64_t, int64_t, int64_t, Sum>(src, dst, dim, stream);
        case float32: return launchArgReduce<float,   float,   double,  Sum>(src, dst, dim, stream);
        case float64: return launchArgReduce<double,  double,  double,  Sum>(src, dst, dim, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status argmean(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case float32: return launchArgReduce<float,  float,  double, Mean>(src, dst, dim, stream);
        case float64: return launchArgReduce<double, double, double, Mean>(src, dst, dim, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cuda
