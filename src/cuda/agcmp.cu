#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/argcmp.cuh"  

namespace {

template<typename S, typename D, typename Cmp>
__global__ void argCompareKernel(
    const S* __restrict__ src, shape_t src_shape, strides_t src_strides,
    D* __restrict__ dst, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, uint8_t dim, S initial_value, size_t ne
) {
    Cmp cmp{};

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < ne;
         idx += blockDim.x * gridDim.x)
    { 
        size_t remaining = idx;
        size_t base_src_offs = 0;

        for (int d = rank - 1; d >= 0; --d) {
            size_t dim_idx = remaining % dst_shape.sizes[d];
            remaining /= dst_shape.sizes[d];
            size_t src_idx = (src_shape.sizes[d] == 1) ? 0 : dim_idx;
            base_src_offs += src_idx * src_strides.sizes[d];
        }

        int64_t best_idx = 0;
        S best_val = initial_value;

        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offs = base_src_offs + i * src_strides.sizes[dim];
            const S val = src[offs];
            if (cmp(val, best_val) || (val == best_val && i < (size_t)best_idx)) {
                best_val = val;
                best_idx = (int64_t)i;
            }
        }

        dst[idx] = static_cast<D>(best_idx);
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

template<typename S, typename Cmp>
status launchArgCompare(const tensor_t* src, tensor_t* dst, uint8_t dim, S init, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t ne = 1;
    for (int i = 0; i < dst->rank; ++i) {
        ne *= dst->shape.sizes[i];
    }

    int threads = 256;
    int blocks = (ne + threads - 1) / threads;

    argCompareKernel<S, int64_t, Cmp><<<blocks, threads, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<int64_t*>(dst->address), dst->shape, dst->strides,
        src->rank, dim, init, ne
    );
    return cudaGetLastError() == cudaSuccess ? SUCCESS : ERROR;
} 

} namespace cuda {

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

}