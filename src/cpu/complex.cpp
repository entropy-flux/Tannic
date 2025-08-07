#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include "cpu/complex.hpp"

namespace {

template<typename Re, typename Im, typename C>
void scalarViewAsCartesianKernel(
    const Re* re_ptr, const Im* im_ptr, C* dst_ptr
) {
    dst_ptr[0] = *re_ptr;
    dst_ptr[1] = *im_ptr;
}    
    
template<typename Re, typename Im, typename C>
void stridedViewAsCartesianKernel(
    const Re* re_ptr, const shape_t& re_shape, const strides_t& re_ne,
    const Im* im_ptr, const shape_t& im_shape, const strides_t& im_ne,
    C* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank
) {  
    size_t cnt[8] = {0}; 
    for (size_t idx = 0;; ++idx) {
        size_t re_offs = 0, im_offs = 0;

        for (uint8_t i = 0; i < rank; ++i) { 
            size_t idx_re = (re_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx_im = (im_shape.sizes[i] == 1) ? 0 : cnt[i];
            
            re_offs += idx_re * re_ne.sizes[i];
            im_offs += idx_im * im_ne.sizes[i];
        }

        dst_ptr[2 * idx]     = re_ptr[re_offs];
        dst_ptr[2 * idx + 1] = im_ptr[im_offs];

        bool done = false;
        for (int i = rank - 1; i >= 0; --i) {
            if (++cnt[i] < dst_shape.sizes[i])
                break;
            if (i == 0)
                done = true;
            cnt[i] = 0;
        }

        if (done) break;
    }
}
 
template<typename Re, typename Im, typename C>
status launchViewAsCartesianKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    if (dst->rank == 0) {
        scalarViewAsCartesianKernel<Re, Im, C>(
            (const Re*)(src0->address), 
            (const Im*)(src1->address),  
            (C*)(dst->address)
        ); 
    } 
    
    else {     
        stridedViewAsCartesianKernel<Re, Im, C>(
            (const Re*)(src0->address), src0->shape, src0->strides,
            (const Im*)(src1->address), src1->shape, src1->strides,
            (C*)(dst->address), dst->shape, dst->strides,
            dst->rank
        ); 
    } 
    return SUCCESS;
}       

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*);      

constexpr static status launchDefaultKernel(const tensor_t*, const tensor_t*, tensor_t*) {
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

} namespace cpu {

status view_as_cartesian(const tensor_t* re, const tensor_t* im, tensor_t* c) {
    return dispatchViewAsCartesian[index(re->dtype, im->dtype)](re, im, c);
}

}