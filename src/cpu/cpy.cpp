#include <array>
#include <complex>
#include <iostream>
#include <cassert>
#include "cpu/cpy.hpp"
#ifndef HAS_FLOAT16
    #if defined(__STDCPP_FLOAT16_T__) && __STDCPP_FLOAT16_T__
        #include <stdfloat>
        using half = std::float16_t;
        using bhalf = std::bfloat16_t;
        #define HAS_FLOAT16 1
    #else 
        #define HAS_FLOAT16 0 
    #endif
#endif 

namespace { 
template<typename S, typename D>
void stridedCpyKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, size_t ne
) {
    size_t cnt[8] = {0};
    for (size_t idx = 0; idx < ne; ++idx) {
        size_t src_offs = 0;
        size_t dst_offs = 0; 

        for (int dim = 0; dim < rank; ++dim) {
            size_t src_coord = (src_shape.sizes[dim] == 1) ? 0 : cnt[dim];
            size_t dst_coord = (dst_shape.sizes[dim] == 1) ? 0 : cnt[dim];

            src_offs += src_coord * src_strides.sizes[dim];
            dst_offs += dst_coord * dst_strides.sizes[dim];
        }
 
        dst_ptr[dst_offs] = src_ptr[src_offs];

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_shape.sizes[dim])
                break;
            cnt[dim] = 0;
        }
    }
}

template<typename S, typename D>
status launchCpyKernel(const tensor_t* src, const tensor_t* dst) {  
    shape_t src_shape; 
    strides_t src_strides; 
    shape_t dst_shape;
    strides_t dst_strides; 

    for (int dim = 0; dim < src->rank; ++dim) {
        src_shape.sizes[dim] = src->shape.sizes[dim];
        src_strides.sizes[dim] = src->strides.sizes[dim];
        dst_shape.sizes[dim] = dst->shape.sizes[dim];
        dst_strides.sizes[dim] = dst->strides.sizes[dim];
    }  
     
    
    stridedCpyKernel<S, D>(
        static_cast<const S*>(src->address), src_shape, src_strides,
        static_cast<D*>(dst->address), dst_shape, dst_strides,
        src->rank, src->size
    );
 
    return SUCCESS;
}

constexpr auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static  status launchDefaultKernel(const tensor_t*, tensor_t*) {
    return UNSUPPORTED_DTYPE;
}; 

using Kernel = status(*)(const tensor_t*, tensor_t*); 

} namespace cpu {

status cpy(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:       return launchCpyKernel<int8_t, int8_t>(src, dst);
        case int16:      return launchCpyKernel<int16_t, int16_t>(src, dst);
        case int32:      return launchCpyKernel<int32_t, int32_t>(src, dst);
        case int64:      return launchCpyKernel<int64_t, int64_t>(src, dst);
#if HAS_FLOAT16
        case float16:    return launchCpyKernel<half, half>(src, dst);
        case bfloat16:   return launchCpyKernel<bhalf, bhalf>(src, dst);
#endif
        case float32:    return launchCpyKernel<float, float>(src, dst);
        case float64:    return launchCpyKernel<double, double>(src, dst);
        case complex64:  return launchCpyKernel<std::complex<float>, std::complex<float>>(src, dst);
        case complex128: return launchCpyKernel<std::complex<double>, std::complex<double>>(src, dst);
        default:         return UNSUPPORTED_DTYPE;
    }
}

}