#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/complex.cuh"

namespace {  

struct Polar {
    template<typename Rho, typename Theta, typename D>
    __device__ __forceinline__ void operator()(Rho&& rho, Theta&& theta, D* out) const {  
        out[0] = rho * cos(theta);
        out[1] = rho * sin(theta); 
    }
}; 

struct Cartesian {
    template<typename Real, typename Imaginary, typename D>
    __device__ __forceinline__ void operator()(Real&& re, Imaginary&& im, D* out) const {  
        out[0] = re;
        out[1] = im; 
    }
};    

template<typename S0, typename S1, typename C, class Coords>
__global__ void scalarComplexViewKernel(
    const S0* src0_ptr, const S1* src1_ptr, C* dst_ptr
) { 
    Coords coords;
    coords(*src0_ptr, *src1_ptr, dst_ptr);
}       

template<typename S0, typename S1, typename D, class Coords>
__global__ void stridedComplexViewKernel(
    const S0* src0_ptr, shape_t src0_shape, strides_t src0_strides,
    const S1* src1_ptr, shape_t src1_shape, strides_t src1_strides,
    D* __restrict__ dst_ptr, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne
) {
    Coords coords;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        size_t cnt[8] = {0};
        size_t remaining = idx;
        for (uint8_t i = rank - 1; i > 0; --i) {
            cnt[i] = remaining % dst_shape.sizes[i];
            remaining /= dst_shape.sizes[i];
        }
        cnt[0] = remaining;

        size_t offs0 = 0, offs1 = 0;
        for (uint8_t i = 0; i < rank; ++i) {
            size_t idx0 = (src0_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx1 = (src1_shape.sizes[i] == 1) ? 0 : cnt[i];
            offs0 += idx0 * src0_strides.sizes[i];
            offs1 += idx1 * src1_strides.sizes[i];
        }  
        coords(src0_ptr[offs0], src1_ptr[offs1], &dst_ptr[2*idx]);
    }
} 

template<typename S0, typename S1, typename C, class Coords>
status launchComplexViewKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    if (dst->rank == 0) {
        scalarComplexViewKernel<S0, S1, C, Coords><<<1, 1, 0, cudaStream>>>(
            (const S0*)(src0->address), 
            (const S1*)(src1->address), 
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

        stridedComplexViewKernel<S0, S1, C, Coords><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S0*)(src0->address), src0->shape, src0->strides,
            (const S1*)(src1->address), src1->shape, src1->strides,
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
    table[index(int8, int8)]     = launchComplexViewKernel<int8_t, int8_t,  float, Cartesian>;
    table[index(int8, int16)]    = launchComplexViewKernel<int8_t, int16_t, float, Cartesian>;
    table[index(int8, int32)]    = launchComplexViewKernel<int8_t, int32_t, float, Cartesian>;
    table[index(int8, int64)]    = launchComplexViewKernel<int8_t, int64_t, float, Cartesian>;

    table[index(int16, int8)]    = launchComplexViewKernel<int16_t, int8_t,  float, Cartesian>;
    table[index(int16, int16)]   = launchComplexViewKernel<int16_t, int16_t, float, Cartesian>;
    table[index(int16, int32)]   = launchComplexViewKernel<int16_t, int32_t, float, Cartesian>;
    table[index(int16, int64)]   = launchComplexViewKernel<int16_t, int64_t, float, Cartesian>;

    table[index(int32, int8)]    = launchComplexViewKernel<int32_t, int8_t,  float, Cartesian>;
    table[index(int32, int16)]   = launchComplexViewKernel<int32_t, int16_t, float, Cartesian>;
    table[index(int32, int32)]   = launchComplexViewKernel<int32_t, int32_t, float, Cartesian>;
    table[index(int32, int64)]   = launchComplexViewKernel<int32_t, int64_t, float, Cartesian>;

    table[index(int64, int8)]    = launchComplexViewKernel<int64_t, int8_t,  float, Cartesian>;
    table[index(int64, int16)]   = launchComplexViewKernel<int64_t, int16_t, float, Cartesian>;
    table[index(int64, int32)]   = launchComplexViewKernel<int64_t, int32_t, float, Cartesian>;
    table[index(int64, int64)]   = launchComplexViewKernel<int64_t, int64_t, float, Cartesian>;

    table[index(int32, float32)] = launchComplexViewKernel<int32_t, float,   float, Cartesian>;
    table[index(float32, int32)] = launchComplexViewKernel<float, int32_t,   float, Cartesian>;
    table[index(int32, float64)] = launchComplexViewKernel<int32_t, double,  double, Cartesian>;
    table[index(float64, int32)] = launchComplexViewKernel<double, int32_t,  double, Cartesian>;

    table[index(float32, float32)] = launchComplexViewKernel<float,  float,  float, Cartesian>;
    table[index(float32, float64)] = launchComplexViewKernel<float,  double, double, Cartesian>;
    table[index(float64, float32)] = launchComplexViewKernel<double, float,  double, Cartesian>;
    table[index(float64, float64)] = launchComplexViewKernel<double, double, double, Cartesian>;
    return table;
}(); 
 
constexpr auto dispatchViewAsPolar = []() {
    std::array<Kernel, index(TYPES, TYPES)> table;
    table.fill(launchDefaultKernel); 
    table[index(float32, float32)] = launchComplexViewKernel<float,  float,  float,  Polar>;
    table[index(float32, float64)] = launchComplexViewKernel<float,  double, double, Polar>;
    table[index(float64, float32)] = launchComplexViewKernel<double, float,  double, Polar>;
    table[index(float64, float64)] = launchComplexViewKernel<double, double, double, Polar>;
    return table;
}();

} namespace cuda {

status view_as_cartesian(const tensor_t* re, const tensor_t* im, tensor_t* c, stream_t stream) {
    return dispatchViewAsCartesian[index(re->dtype, im->dtype)](re, im, c, stream);
}

status view_as_polar(const tensor_t* rho, const tensor_t* theta, tensor_t* c, stream_t stream) {
    return dispatchViewAsPolar[index(rho->dtype, theta->dtype)](rho, theta, c, stream);
}

} // namespace cuda