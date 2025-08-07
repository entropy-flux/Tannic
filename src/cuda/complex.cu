#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/complex.cuh"

namespace {

template<typename Re, typename Im, typename D>
__global__ void scalariewAsCartesianKernel(
    const Re* re_ptr, const Im* im_ptr, D* dst_ptr
) {
    dst_ptr[0] = *re_ptr;
    dst_ptr[1] = *im_ptr;
}    

template<typename Re, typename Im, typename D>
__global__ void stridedViewAsCartesianKernel(
    const Re* re_ptr, shape_t re_shape, strides_t re_strides,
    const Im* im_ptr, shape_t im_shape, strides_t im_strides,
    D* __restrict__ dst_ptr, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne
) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        size_t cnt[8] = {0};
        size_t remaining = idx;
 
        for (uint8_t i = rank - 1; i > 0; --i) {
            cnt[i] = remaining % dst_shape.sizes[i];
            remaining /= dst_shape.sizes[i];
        }
        cnt[0] = remaining;
 
        size_t re_offs = 0, im_offs = 0;
        for (uint8_t i = 0; i < rank; ++i) {
            size_t idx_re = (re_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx_im = (im_shape.sizes[i] == 1) ? 0 : cnt[i];

            re_offs += idx_re * re_strides.sizes[i];
            im_offs += idx_im * im_strides.sizes[i];
        }

        dst_ptr[2 * idx]     = re_ptr[re_offs];
        dst_ptr[2 * idx + 1] = im_ptr[im_offs];
    }
}
 
template<typename Re, typename Im, typename C>
status launchViewAsCartesianKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    if (dst->rank == 0) {
        scalariewAsCartesianKernel<Re, Im, C><<<1, 1, 0, cudaStream>>>(
            (const Re*)(src0->address), 
            (const Im*)(src1->address), 
            (C*)(dst->address)
        );   
    } 
    
    else {     
        size_t ne = 1;
        for (uint8_t dim = 0; dim < dst->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }
        
        int blockSize = 256;
        int gridSize = (ne + blockSize - 1) / blockSize;

        stridedViewAsCartesianKernel<Re, Im, C><<<gridSize, blockSize, 0, cudaStream>>>(
            (const Re*)(src0->address), src0->shape, src0->strides,
            (const Im*)(src1->address), src1->shape, src1->strides,
            (C*)(dst->address), dst->shape, dst->strides,
            dst->rank, ne
        ); 
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

constexpr auto dispatchViewAsCartesian = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(launchDefaultKernel); 
    table[index(int8, int8)]     = launchViewAsCartesianKernel<int8_t, int8_t,  float>;
    table[index(int8, int16)]    = launchViewAsCartesianKernel<int8_t, int16_t, float>;
    table[index(int8, int32)]    = launchViewAsCartesianKernel<int8_t, int32_t, float>;
    table[index(int8, int64)]    = launchViewAsCartesianKernel<int8_t, int64_t, float>;

    table[index(int16, int8)]    = launchViewAsCartesianKernel<int16_t, int8_t,  float>;
    table[index(int16, int16)]   = launchViewAsCartesianKernel<int16_t, int16_t, float>;
    table[index(int16, int32)]   = launchViewAsCartesianKernel<int16_t, int32_t, float>;
    table[index(int16, int64)]   = launchViewAsCartesianKernel<int16_t, int64_t, float>;

    table[index(int32, int8)]    = launchViewAsCartesianKernel<int32_t, int8_t,  float>;
    table[index(int32, int16)]   = launchViewAsCartesianKernel<int32_t, int16_t, float>;
    table[index(int32, int32)]   = launchViewAsCartesianKernel<int32_t, int32_t, float>;
    table[index(int32, int64)]   = launchViewAsCartesianKernel<int32_t, int64_t, float>;

    table[index(int64, int8)]    = launchViewAsCartesianKernel<int64_t, int8_t,  float>;
    table[index(int64, int16)]   = launchViewAsCartesianKernel<int64_t, int16_t, float>;
    table[index(int64, int32)]   = launchViewAsCartesianKernel<int64_t, int32_t, float>;
    table[index(int64, int64)]   = launchViewAsCartesianKernel<int64_t, int64_t, float>;

    table[index(int32, float32)] = launchViewAsCartesianKernel<int32_t, float,   float>;
    table[index(float32, int32)] = launchViewAsCartesianKernel<float, int32_t,   float>;
    table[index(int32, float64)] = launchViewAsCartesianKernel<int32_t, double,  double>;
    table[index(float64, int32)] = launchViewAsCartesianKernel<double, int32_t,  double>;

    table[index(float32, float32)] = launchViewAsCartesianKernel<float,  float,  float>;
    table[index(float32, float64)] = launchViewAsCartesianKernel<float,  double, double>;
    table[index(float64, float32)] = launchViewAsCartesianKernel<double, float,  double>;
    table[index(float64, float64)] = launchViewAsCartesianKernel<double, double, double>;
    return table;
}(); 

} namespace cuda {

status view_as_cartesian(const tensor_t* re, const tensor_t* im, tensor_t* c, stream_t stream) {
    return dispatchViewAsCartesian[index(re->dtype, im->dtype)](re, im, c, stream);
}

} // namespace cuda