#include <array>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda/outer.cuh"

namespace {
 
template<typename S1, typename S2, typename D>
__global__ void vectorOuterKernel(
    const S1* src1_ptr, size_t n1,
    const S2* src2_ptr, size_t n2,
    D* dst_ptr
) {
    // Calculate global thread indices for 2D grid
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n1 && j < n2) {
        dst_ptr[i * n2 + j] = static_cast<D>(src1_ptr[i]) * static_cast<D>(src2_ptr[j]); 
    }
}
 
template<typename S1, typename S2, typename D>
status launchOuterKernel(const tensor_t* src1, const tensor_t* src2, tensor_t* dst, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t n1 = src1->shape.sizes[0];
    size_t n2 = src2->shape.sizes[0];
 
    dim3 blockDim(16, 16);
    dim3 gridDim((n2 + blockDim.x - 1) / blockDim.x, (n1 + blockDim.y - 1) / blockDim.y);

    vectorOuterKernel<S1, S2, D><<<gridDim, blockDim, 0, cudaStream>>>(
        (const S1*)src1->address, n1,
        (const S2*)src2->address, n2,
        (D*)dst->address
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ERROR;
    } 

    return SUCCESS;
}
  
using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);      

constexpr static status launchDefaultKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {
    return UNSUPPORTED_DTYPE;
}; 

constexpr static inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  

constexpr auto dispatchOuter = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(launchDefaultKernel); 
    table[index(int8, int8)]     = launchOuterKernel<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = launchOuterKernel<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = launchOuterKernel<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = launchOuterKernel<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = launchOuterKernel<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = launchOuterKernel<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = launchOuterKernel<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = launchOuterKernel<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = launchOuterKernel<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = launchOuterKernel<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = launchOuterKernel<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = launchOuterKernel<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = launchOuterKernel<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = launchOuterKernel<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = launchOuterKernel<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = launchOuterKernel<int64_t, int64_t, int64_t>;
    
    table[index(float16, float16)] = launchOuterKernel<__half, __half, __half>;
    table[index(float16, float32)] = launchOuterKernel<__half, float, float>;
    table[index(float16, float64)] = launchOuterKernel<__half, double, double>;
    table[index(float32, float16)] = launchOuterKernel<float, __half, float>;
    table[index(float64, float16)] = launchOuterKernel<double, __half, double>;
    
    table[index(int8, float16)]    = launchOuterKernel<int8_t, __half, __half>;
    table[index(int16, float16)]   = launchOuterKernel<int16_t, __half, __half>;
    table[index(int32, float16)]   = launchOuterKernel<int32_t, __half, float>;
    table[index(int64, float16)]   = launchOuterKernel<int64_t, __half, double>;
    table[index(float16, int8)]    = launchOuterKernel<__half, int8_t, __half>;
    table[index(float16, int16)]   = launchOuterKernel<__half, int16_t, __half>;
    table[index(float16, int32)]   = launchOuterKernel<__half, int32_t, float>;
    table[index(float16, int64)]   = launchOuterKernel<__half, int64_t, double>;

    table[index(int32, float32)] = launchOuterKernel<int32_t, float, float>;
    table[index(float32, int32)] = launchOuterKernel<float, int32_t, float>;
    table[index(int32, float64)] = launchOuterKernel<int32_t, double, double>;
    table[index(float64, int32)] = launchOuterKernel<double, int32_t, double>;

    table[index(float32, float32)] = launchOuterKernel<float, float, float>;
    table[index(float32, float64)] = launchOuterKernel<float, double, double>;
    table[index(float64, float32)] = launchOuterKernel<double, float, double>;
    table[index(float64, float64)] = launchOuterKernel<double, double, double>;
    return table;
}(); 

} namespace cuda {

status outer(tensor_t const* src0, tensor_t const* src1, tensor_t* dst, stream_t stream) {
    return dispatchOuter[index(src0->dtype, src1->dtype)](src0, src1, dst, stream);
}

}