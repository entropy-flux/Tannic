#include <cstdint>
#include <array> 
#include <cuda_runtime.h> 
#include "cuda/argred.cuh"

namespace {
 
template<typename S, typename D, typename Op>
__global__ void stridedArgCompareKernel(
    const S* __restrict__ src, shape_t src_shape, strides_t src_strides,
    D* __restrict__ dst, 
    uint8_t rank, uint8_t ax, S initial_value, size_t ne
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
        
        size_t remaining = idx;
        size_t base_src_offs = 0;

        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
            const size_t size_d = (d == ax) ? 1 : src_shape.sizes[d];
            const size_t idx_d  = (size_d > 1) ? (remaining % size_d) : 0;
            if (size_d > 1) remaining /= size_d;

            const size_t src_idx = (src_shape.sizes[d] == 1) ? 0 : idx_d;
            base_src_offs += src_idx * src_strides.sizes[d];
        }

        int64_t best_idx = 0;
        S best_val = initial_value;

        for (size_t i = 0; i < src_shape.sizes[ax]; ++i) {
            const size_t offs = base_src_offs + i * src_strides.sizes[ax];
            const S val = src[offs];
            if (cmp(val, best_val) || (val == best_val && i < static_cast<size_t>(best_idx))) {
                best_val = val;
                best_idx = static_cast<int64_t>(i);
            }
        }

        dst[idx] = static_cast<D>(best_idx);
    }
}


template<typename S, typename D, typename A, class Op>
__global__ void stridedArgReduceKernel(
    const S* __restrict__ src_ptr, const shape_t src_shape, const strides_t src_strides,
    D* __restrict__ dst_ptr,
    uint8_t rank, uint8_t ax, size_t ne
) {
    Op op;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
 
        size_t cnt[8] = {0};
        size_t remaining = idx;
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (dim == ax) continue;
            const size_t sz = src_shape.sizes[dim];
            cnt[dim] = (sz ? (remaining % sz) : 0);
            remaining = (sz ? (remaining / sz) : 0);
        }
 
        A accum = A(0);
        const size_t reduceN = src_shape.sizes[ax];
        for (size_t i = 0; i < reduceN; ++i) {
            size_t offset = 0;
            for (int dim = 0; dim < rank; ++dim) {
                const size_t idx_val = (dim == ax) ? i : cnt[dim];
                offset += idx_val * src_strides.sizes[dim]; 
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
status launchArgCompare(const tensor_t* src, tensor_t* dst, uint8_t ax, S init, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t ne = dst->size;

    int threads = 256;
    int blocks = (ne + threads - 1) / threads; 

    shape_t src_shape;
    strides_t src_strides;
    for (uint8_t dim = 0; dim < src->rank; dim++) {
        src_shape.sizes[dim] = src->shape.sizes[dim];
        src_strides.sizes[dim] = src->strides.sizes[dim];
    } 

    stridedArgCompareKernel<S, int64_t, Op><<<blocks, threads, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src_shape, src_strides,
        reinterpret_cast<int64_t*>(dst->address), 
        src->rank, ax, init, ne
    );
    return cudaGetLastError() == cudaSuccess ? SUCCESS : ERROR;
} 

template<typename S, typename D, typename A, class Op>
status launchArgReduce(const tensor_t* src, tensor_t* dst, uint8_t ax, stream_t stream) {
    if (src->rank == 0 || ax >= src->rank) return SUCCESS;  
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t ne = src->size / src->shape.sizes[ax];
    const int blockSize = 256;
    const int gridSize  = static_cast<int>((ne + blockSize - 1) / blockSize); 

    shape_t src_shape;
    strides_t src_strides;
    for (uint8_t dim = 0; dim < src->rank; dim++) {
        src_shape.sizes[dim] = src->shape.sizes[dim];
        src_strides.sizes[dim] = src->strides.sizes[dim];
    } 

    stridedArgReduceKernel<S, D, A, Op><<<gridSize, blockSize, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src_shape, src_strides,
        reinterpret_cast<D*>(dst->address), 
        src->rank, ax, ne
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
