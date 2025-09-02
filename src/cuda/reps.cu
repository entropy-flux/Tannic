#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <cstdint>
#include <array> 
#include "cuda/reps.cuh"

namespace {

template<typename T>
__global__ void stridedRepeatsKernel(
    const T* __restrict__ src_ptr,
    const shape_t src_shape,
    const strides_t src_strides,
    T* __restrict__ dst_ptr,
    uint8_t rank,
    int dim,
    int reps,
    size_t outer_ne,
    size_t inner_ne
) {
    size_t dim_size = src_shape.sizes[dim];
    size_t src_stride = src_strides.sizes[dim];

    size_t total_work = outer_ne * dim_size * reps * inner_ne;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_work) return;
 
    size_t inner_k = idx % inner_ne; idx /= inner_ne;
    int r = idx % reps; idx /= reps;
    int j = idx % dim_size; idx /= dim_size;
    size_t outer_idx = idx;
 
    size_t outer_offset = 0;
    size_t tmp = outer_idx;
    for (int i = dim - 1; i >= 0; --i) {
        size_t coord = tmp % src_shape.sizes[i];
        tmp /= src_shape.sizes[i];
        outer_offset += coord * src_strides.sizes[i];
    }

    const T* src_block = src_ptr + outer_offset + j * src_stride;
 
    dst_ptr[(outer_idx * dim_size * reps + j * reps + r) * inner_ne + inner_k] = src_block[inner_k];
}

template<typename T>
status launchRepeatsKernel(const tensor_t* src, tensor_t* dst, int axis, int reps, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t outer_ne = 1;
    for (int i = 0; i < axis; ++i) {
        outer_ne *= src->shape.sizes[i];
    }
    size_t inner_ne = 1;
    for (int i = axis + 1; i < src->rank; ++i) {
        inner_ne *= src->shape.sizes[i];
    }

    size_t dim_size = src->shape.sizes[axis];
    size_t total_work = outer_ne * dim_size * reps * inner_ne;

    int threads = 256;
    int blocks = (total_work + threads - 1) / threads;

    stridedRepeatsKernel<T><<<blocks, threads, 0, cudaStream>>>(
        static_cast<const T*>(src->address),
        src->shape,
        src->strides,
        static_cast<T*>(dst->address),
        dst->rank,
        axis,
        reps,
        outer_ne,
        inner_ne
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        return ERROR; 
    else
        return SUCCESS;
} 

constexpr static status launchDefaultKenel(const tensor_t* src, tensor_t* dst, int axis, int reps, stream_t stream) {
    return UNSUPPORTED_DTYPE;
}

constexpr static int index(type dtype) {
    return static_cast<int>(dtype);
} 

using Kernel = status(*)(const tensor_t*, tensor_t*, int, int, stream_t);      

constexpr auto dispatchRepeatsKernel = []() {  
    std::array<Kernel, index(TYPES)> table; table.fill(launchDefaultKenel);
    table[index(int8)] = launchRepeatsKernel  <int8_t>;
    table[index(int16)] = launchRepeatsKernel <int16_t>;
    table[index(int32)] = launchRepeatsKernel <int32_t>;
    table[index(int64)] = launchRepeatsKernel <int64_t>;
    table[index(float16)] = launchRepeatsKernel<__half>;
    table[index(float32)] = launchRepeatsKernel<float>;
    table[index(float64)] = launchRepeatsKernel<double>;
    return table;
}();  

}  namespace cuda {

status repeat(const tensor_t* src, tensor_t* dst, int dim, int reps, stream_t stream) {
    return dispatchRepeatsKernel[index(dst->dtype)](src, dst, dim, reps, stream);
}

}