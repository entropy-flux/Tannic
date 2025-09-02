#include <array> 
#include <cuda_fp16.h> 
#include "cuda/concat.cuh"  

namespace {

template<typename T>
__global__ void stridedConcatKernel(
    const T* srcA_ptr, const shape_t srcA_shape, const strides_t srcA_strides,
    const T* srcB_ptr, const shape_t srcB_shape, const strides_t srcB_strides,
    T* dst_ptr,
    uint8_t rank,
    int dim,
    size_t total_ne
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_ne) return;
 
    size_t coord[8];
    size_t tmp = idx;
    for (int i = rank - 1; i >= 0; --i) {
        coord[i] = tmp % (srcA_shape.sizes[i] + (i == dim ? srcB_shape.sizes[i] : 0));
        tmp /= (srcA_shape.sizes[i] + (i == dim ? srcB_shape.sizes[i] : 0));
    }

    const T* src_base;
    const strides_t* src_strides;
    size_t src_offset = 0;

    if (coord[dim] < srcA_shape.sizes[dim]) { 
        src_base = srcA_ptr;
        src_strides = &srcA_strides;
        for (int i = 0; i < rank; ++i) {
            src_offset += coord[i] * src_strides->sizes[i];
        }
    } else { 
        coord[dim] -= srcA_shape.sizes[dim];
        src_base = srcB_ptr;
        src_strides = &srcB_strides;
        for (int i = 0; i < rank; ++i) {
            src_offset += coord[i] * src_strides->sizes[i];
        }
    } 
    dst_ptr[idx] = src_base[src_offset];
}
 
template<typename T>
status launchConcatKernel(const tensor_t* srcA, const tensor_t* srcB, tensor_t* dst, stream_t stream, int axis) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t total_ne = 1;
    for (int i = 0; i < dst->rank; ++i) {
        total_ne *= dst->shape.sizes[i];
    }

    dim3 block(256);
    dim3 grid((total_ne + block.x - 1) / block.x);

    stridedConcatKernel<T><<<grid, block, 0, cudaStream>>>(
        static_cast<const T*>(srcA->address), srcA->shape, srcA->strides,
        static_cast<const T*>(srcB->address), srcB->shape, srcB->strides,
        static_cast<T*>(dst->address), dst->rank, axis, total_ne
    );

    return cudaGetLastError() == cudaSuccess ? SUCCESS : ERROR;
};

constexpr static status launchDefaultKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int) {
    return UNSUPPORTED_DTYPE;
}

constexpr static int index(type dtype) {
    return static_cast<int>(dtype);
}

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t, int);

constexpr auto dispatchConcatKernel = []() {
    std::array<Kernel, index(TYPES)> table;
    table.fill(launchDefaultKernel);
    table[index(int8)]     = launchConcatKernel<int8_t>;
    table[index(int16)]    = launchConcatKernel<int16_t>;
    table[index(int32)]    = launchConcatKernel<int32_t>;
    table[index(int64)]    = launchConcatKernel<int64_t>;
    table[index(float16)]  = launchConcatKernel<__half>;
    table[index(float32)]  = launchConcatKernel<float>;
    table[index(float64)]  = launchConcatKernel<double>;
    return table;
}();

} namespace cuda {

status concat(const tensor_t* srcA, const tensor_t* srcB, tensor_t* dst, stream_t stream, int dim) {
    return dispatchConcatKernel[index(dst->dtype)](srcA, srcB, dst, stream, dim);
}

} 