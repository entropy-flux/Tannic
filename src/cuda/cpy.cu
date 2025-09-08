#include "cuda/cpy.cuh" 
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <cuda_fp16.h>  
#include <cuda_bf16.h> 

namespace {

template<typename S, typename D>
__global__ void stridedCpyKernel(
    const S* __restrict__ src_ptr, shape_t src_shape, strides_t src_strides,
    D* __restrict__ dst_ptr, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank,
    size_t ne
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        size_t tmp = idx;
        size_t src_offs = 0;
        size_t dst_offs = 0;
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t coord = tmp % dst_shape.sizes[dim];
            tmp /= dst_shape.sizes[dim];

            size_t src_coord = (src_shape.sizes[dim] == 1) ? 0 : coord;
            size_t dst_coord = (dst_shape.sizes[dim] == 1) ? 0 : coord;

            src_offs += src_coord * src_strides.sizes[dim];
            dst_offs += dst_coord * dst_strides.sizes[dim];
        }

        dst_ptr[dst_offs] = static_cast<D>(src_ptr[src_offs]);
    }
}

template<typename S, typename D>
status launchCpyKernel(const tensor_t* src, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    dim3 threads(256);
    dim3 blocks((dst->size + threads.x - 1) / threads.x);

    stridedCpyKernel<S,D><<<blocks, threads, 0, cudaStream>>>(
        static_cast<const S*>(src->address), src->shape, src->strides,
        static_cast<D*>(dst->address), dst->shape, dst->strides,
        dst->rank, dst->size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return ERROR;
    return SUCCESS;
}

} namespace cuda {

status cpy(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case int8:       return launchCpyKernel<int8_t, int8_t>(src, dst, stream);
        case int16:      return launchCpyKernel<int16_t, int16_t>(src, dst, stream);
        case int32:      return launchCpyKernel<int32_t, int32_t>(src, dst, stream);
        case int64:      return launchCpyKernel<int64_t, int64_t>(src, dst, stream); 
        case float16:    return launchCpyKernel<__half, __half>(src, dst, stream);
        case bfloat16:   return launchCpyKernel<__nv_bfloat16, __nv_bfloat16>(src, dst, stream); 
        case float32:    return launchCpyKernel<float, float>(src, dst, stream);
        case float64:    return launchCpyKernel<double, double>(src, dst, stream);
        case complex64:  return launchCpyKernel<thrust::complex<float>, thrust::complex<float>>(src, dst, stream);
        case complex128: return launchCpyKernel<thrust::complex<double>, thrust::complex<double>>(src, dst, stream);
        default:         return UNSUPPORTED_DTYPE;
    }
}

} // namespace