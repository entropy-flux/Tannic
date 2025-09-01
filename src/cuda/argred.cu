#include <cstdint>
#include <array> 
#include <cuda_runtime.h> 
#include "cuda/argred.cuh"

namespace {
 
template<typename S, typename D, typename Op>
__global__ void argCompareKernel(
    const S* __restrict__ src, shape_t src_shape, strides_t src_strides,
    D* __restrict__ dst, 
    uint8_t rank, uint8_t dim, S initial_value, size_t ne
) {
    Op cmp{};
 
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < ne;
         idx += blockDim.x * gridDim.x)
    {
        if (rank == 0) {
            if (idx == 0) dst[0] = static_cast<D>(0);
            continue;
        }

        // Map flat idx -> multi-index using src_shape with reduced dim treated as 1
        size_t remaining = idx;
        size_t base_src_offs = 0;

        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
            const size_t size_d = (d == dim) ? 1 : src_shape.sizes[d];
            const size_t idx_d  = (size_d > 1) ? (remaining % size_d) : 0;
            if (size_d > 1) remaining /= size_d;

            // If the source is broadcast (size 1), always index 0
            const size_t src_idx = (src_shape.sizes[d] == 1) ? 0 : idx_d;
            base_src_offs += src_idx * src_strides.sizes[d];
        }

        int64_t best_idx = 0;
        S best_val = initial_value;

        // Scan along the reduced dimension
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            const size_t offs = base_src_offs + i * src_strides.sizes[dim];
            const S val = src[offs];
            if (cmp(val, best_val) || (val == best_val && i < static_cast<size_t>(best_idx))) {
                best_val = val;
                best_idx = static_cast<int64_t>(i);
            }
        }

        // Assumes dst is contiguous; if it's not, compute dst offset with dst_strides.
        dst[idx] = static_cast<D>(best_idx);
    }
}


template<typename S, typename D, typename A, class Op>
__global__ void argReduceKernel(
    const S* __restrict__ src_ptr, const shape_t src_shape, const strides_t src_strides,
    D* __restrict__ dst_ptr,
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

struct GE {
    template<class A, class B>
    __device__  __forceinline__ bool operator()(A&& a, B&& b) const noexcept {
        return a > b;
    }
};

struct LE {
    template<class A, class B>
    __device__ __forceinline__ bool operator()(A&& a, B&& b) const noexcept {
        return a < b;
    }
}; 

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
 
template<typename S, typename Op>
status launchArgCompare(const tensor_t* src, tensor_t* dst, uint8_t dim, S init, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t ne = dst->size;

    int threads = 256;
    int blocks = (ne + threads - 1) / threads;

    argCompareKernel<S, int64_t, Op><<<blocks, threads, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<int64_t*>(dst->address), 
        src->rank, dim, init, ne
    );
    return cudaGetLastError() == cudaSuccess ? SUCCESS : ERROR;
} 

template<typename S, typename D, typename A, class Op>
status launchArgReduce(const tensor_t* src, tensor_t* dst, uint8_t dim, stream_t stream) {
    if (src->rank == 0 || dim >= src->rank) return SUCCESS; // or an error code if you prefer
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    
    size_t ne = src->size / src->shape.sizes[dim];
    const int blockSize = 256;
    const int gridSize  = static_cast<int>((ne + blockSize - 1) / blockSize);

    argReduceKernel<S, D, A, Op><<<gridSize, blockSize, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<D*>(dst->address), 
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

status argmax(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgCompare<int8_t, GE>(src, dst, dim, std::numeric_limits<int8_t>::lowest(), stream);
        case int16:   return launchArgCompare<int16_t, GE>(src, dst, dim, std::numeric_limits<int16_t>::lowest(), stream);
        case int32:   return launchArgCompare<int32_t, GE>(src, dst, dim, std::numeric_limits<int32_t>::lowest(), stream);
        case int64:   return launchArgCompare<int64_t, GE>(src, dst, dim, std::numeric_limits<int64_t>::lowest(), stream);
        case float32: return launchArgCompare<float, GE>(src, dst, dim, -std::numeric_limits<float>::infinity(), stream);
        case float64: return launchArgCompare<double, GE>(src, dst, dim, -std::numeric_limits<double>::infinity(), stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status argmin(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgCompare<int8_t, LE>(src, dst, dim, std::numeric_limits<int8_t>::max(), stream);
        case int16:   return launchArgCompare<int16_t, LE>(src, dst, dim, std::numeric_limits<int16_t>::max(), stream);
        case int32:   return launchArgCompare<int32_t, LE>(src, dst, dim, std::numeric_limits<int32_t>::max(), stream);
        case int64:   return launchArgCompare<int64_t, LE>(src, dst, dim, std::numeric_limits<int64_t>::max(), stream);
        case float32: return launchArgCompare<float, LE>(src, dst, dim, std::numeric_limits<float>::infinity(), stream);
        case float64: return launchArgCompare<double, LE>(src, dst, dim, std::numeric_limits<double>::infinity(), stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}


} // namespace cuda
