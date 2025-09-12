#include <cuda_runtime.h>
#include <cuda_fp16.h>     
#include <cuda_bf16.h>       
#include "triang.cuh"

namespace {

struct Upper {
    int k;
    Upper(int diag = 0) : k(diag) {}

    template<typename T>
    __device__ __forceinline__ T operator()(T value, size_t i, size_t j) const noexcept {
        return (j >= i + k) ? value : T(0);
    }

    __device__ __forceinline__ __half operator()(__half value, size_t i, size_t j) const noexcept {
        return (j >= i + k) ? value : __float2half(0.0f);
    }

    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 value, size_t i, size_t j) const noexcept {
        return (j >= i + k) ? value : __float2bfloat16(0.0f);
    }
};

struct Lower {
    int k; 
    Lower(int diag = 0) : k(diag) {}

    template<typename T>
    __device__ __forceinline__ T operator()(T value, size_t i, size_t j) const noexcept {
        return (j <= i + k) ? value : T(0);
    }

    __device__ __forceinline__ __half operator()(__half value, size_t i, size_t j) const noexcept {
        return (j <= i + k) ? value : __float2half(0.0f);
    }

    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 value, size_t i, size_t j) const noexcept {
        return (j <= i + k) ? value : __float2bfloat16(0.0f);
    }
};
 
template<typename T, class Fn>
__global__ void contiguousTriangularKernel(const T* src, T* dst, size_t rows, size_t cols, Fn tri) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        dst[i * cols + j] = tri(src[i * cols + j], i, j);
    }
}
 
template<typename T, class Fn>
__global__ void stridedTriangularKernel(
    const T* src, const shape_t src_shape, const strides_t src_strides,
    T* dst, const shape_t dst_shape, const strides_t dst_strides,
    Fn tri
) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < src_shape.sizes[0] && j < src_shape.sizes[1]) {
        size_t src_offset = i * src_strides.sizes[0] + j * src_strides.sizes[1];
        size_t dst_offset = i * dst_strides.sizes[0] + j * dst_strides.sizes[1];
        dst[dst_offset] = tri(src[src_offset], i, j);
    }
}

template<typename T, class Fn>
status launchTriangularKernel(const tensor_t* src, tensor_t* dst, Fn tri, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    if (src->rank != 2) return ERROR;

    size_t rows = src->shape.sizes[0];
    size_t cols = src->shape.sizes[1];

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    if (src->layout == CONTIGUOUS) {
        contiguousTriangularKernel<T, Fn><<<grid, block , 0, cudaStream>>>(
            (const T*)src->address, (T*)dst->address, rows, cols, tri
        );
    } else { 
        stridedTriangularKernel<T, Fn><<<grid, block, 0, cudaStream>>>(
            (const T*)src->address, src->shape, src->strides,
            (T*)dst->address, dst->shape, dst->strides,
            tri
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ERROR;

    return SUCCESS;
}

} // namespace

namespace cuda {
 
status triu(const tensor_t* src, tensor_t* dst, stream_t stream, int k) {
    Upper tri(k);
    switch (src->dtype) {
        case int32:     return launchTriangularKernel<int32_t>(src, dst, tri, stream);
        case int64:     return launchTriangularKernel<int64_t>(src, dst, tri, stream);
        case float16:   return launchTriangularKernel<__half>(src, dst, tri, stream);
        case bfloat16:  return launchTriangularKernel<__nv_bfloat16>(src, dst, tri, stream);
        case float32:   return launchTriangularKernel<float>(src, dst, tri, stream);
        case float64:   return launchTriangularKernel<double>(src, dst, tri, stream);  
        default:        return UNSUPPORTED_DTYPE;
    }
}

status tril(const tensor_t* src, tensor_t* dst, stream_t stream, int k) {
    Lower tri(k);
    switch (src->dtype) {
        case int32:     return launchTriangularKernel<int32_t>(src, dst, tri, stream);
        case int64:     return launchTriangularKernel<int64_t>(src, dst, tri, stream);
        case float16:   return launchTriangularKernel<__half>(src, dst, tri, stream);
        case bfloat16:  return launchTriangularKernel<__nv_bfloat16>(src, dst, tri, stream);
        case float32:   return launchTriangularKernel<float>(src, dst, tri, stream);
        case float64:   return launchTriangularKernel<double>(src, dst, tri, stream);   
        default:        return UNSUPPORTED_DTYPE;
    }
}

} // namespace cuda
