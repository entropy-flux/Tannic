#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <cstdint>
#include <array> 
#include "cuda/reps.cuh"

namespace {

template<typename T>
__global__ void stridedRepeatsKernel(
    const T* __restrict__ src_ptr,
    shape_t src_shape,
    strides_t src_strides,
    T* __restrict__ dst_ptr,
    uint8_t rank,
    int axis,
    int reps,
    size_t outer_ne,
    size_t inner_ne
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_work = outer_ne * src_shape.sizes[axis] * reps * inner_ne;

    if (idx >= total_work) return;

    size_t tmp = idx;
    size_t inner_idx = tmp % inner_ne;
    tmp /= inner_ne; 
    tmp /= reps;
    size_t j = tmp % src_shape.sizes[axis];
    size_t outer_idx = tmp / src_shape.sizes[axis];
 
    size_t outer_offset = 0;
    size_t tmp_outer = outer_idx;
    for (int i = axis - 1; i >= 0; --i) {
        size_t coord = tmp_outer % src_shape.sizes[i];
        tmp_outer /= src_shape.sizes[i];
        outer_offset += coord * src_strides.sizes[i];
    }
 
    size_t inner_offset = 0;
    size_t tmp_inner = inner_idx;
    for (int d = rank - 1; d > axis; --d) {
        size_t coord = tmp_inner % src_shape.sizes[d];
        tmp_inner /= src_shape.sizes[d];
        inner_offset += coord * src_strides.sizes[d];
    }

    const T* src_block = src_ptr + outer_offset + j * src_strides.sizes[axis];
    dst_ptr[idx] = src_block[inner_offset];
}

template<typename T>
status launchRepeatsKernel(const tensor_t* src, tensor_t* dst, int axis, int reps, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    shape_t src_shape;
    strides_t src_strides;
    size_t inner_ne = 1;
    size_t outer_ne = 1;
    for(uint8_t dim = 0; dim < src->rank; ++dim) {
        src_shape.sizes[dim] = src->shape.sizes[dim];
        src_strides.sizes[dim] =src->strides.sizes[dim];
        if(dim < axis) {
            outer_ne *= src->shape.sizes[dim];
        } 
        
        else if(dim > axis) {
            inner_ne *= src->shape.sizes[dim]; 
        } 
    }


    size_t dim_size = src->shape.sizes[axis];
    size_t total_work = outer_ne * dim_size * reps * inner_ne;

    int threads = 256;
    int blocks = (total_work + threads - 1) / threads;

    stridedRepeatsKernel<T><<<blocks, threads, 0, cudaStream>>>(
        static_cast<const T*>(src->address),
        src_shape,
        src_strides,
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