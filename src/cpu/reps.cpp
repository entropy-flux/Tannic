#include <cstring>   
#include <array>
#include "cpu/reps.hpp"
#ifndef HAS_FLOAT16
    #if defined(__STDCPP_FLOAT16_T__) && __STDCPP_FLOAT16_T__
        #include <stdfloat>
        using half = std::float16_t;
        #define HAS_FLOAT16 1
    #else 
        #define HAS_FLOAT16 0 
        struct half_placeholder { float value; };
        using half = half_placeholder;
    #endif
#endif
 

namespace {

template<typename T>
void stridedRepeatsKernel(
    const T* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    T* dst_ptr,
    uint8_t rank,
    int dim,        
    int reps 
) {
    size_t outer_ne = 1;
    for (int i = 0; i < dim; ++i) {
        outer_ne *= src_shape.sizes[i];
    }

    size_t inner_ne = 1;
    for (int i = dim + 1; i < rank; ++i) {
        inner_ne *= src_shape.sizes[i];
    }

    size_t dim_size = src_shape.sizes[dim];
    size_t src_stride = src_strides.sizes[dim];
 
    for (size_t outer_idx = 0; outer_idx < outer_ne; ++outer_idx) {
        size_t outer_offset = 0;
        {
            size_t tmp = outer_idx;
            for (int i = dim - 1; i >= 0; --i) {
                size_t coord = tmp % src_shape.sizes[i];
                tmp /= src_shape.sizes[i];
                outer_offset += coord * src_strides.sizes[i];
            }
        }
 
        for (size_t j = 0; j < dim_size; ++j) {
            const T* src_block = src_ptr + outer_offset + j * src_stride;
            for (int r = 0; r < reps; ++r) {
                for (size_t k = 0; k < inner_ne; ++k) {
                    dst_ptr[k] = src_block[k];
                }
                dst_ptr += inner_ne;
            }
        }
    }
} 

template<typename T>
status launchRepeatsKernel(const tensor_t* src, tensor_t* dst, int axis, int reps) {
    stridedRepeatsKernel<T>(
        static_cast<const T*>(src->address), src->shape, src->strides,
        static_cast<T*>(dst->address), dst->rank, axis, reps
    );
    return SUCCESS;
};

constexpr static status launchDefaultKenel(const tensor_t* src, tensor_t* dst, int axis, int reps) {
    return UNSUPPORTED_DTYPE;
}

constexpr static int index(type dtype) {
    return static_cast<int>(dtype);
} 

using Kernel = status(*)( const tensor_t*, tensor_t*, int, int);      

constexpr auto dispatchRepeatsKernel = []() {  
    std::array<Kernel, index(TYPES)> table; table.fill(launchDefaultKenel);
    table[index(int8)] = launchRepeatsKernel<int8_t>;
    table[index(int16)] = launchRepeatsKernel<int16_t>;
    table[index(int32)] = launchRepeatsKernel<int32_t>;
    table[index(int64)] = launchRepeatsKernel<int64_t>;
#if HAS_FLOAT16
    table[index(float16)] = launchRepeatsKernel<half>;
#endif
    table[index(float32)] = launchRepeatsKernel<float>;
    table[index(float64)] = launchRepeatsKernel<double>;
    return table;
}();  

}  namespace cpu {

status repeat(const tensor_t* src, tensor_t* dst, int dim, int reps) {
    return dispatchRepeatsKernel[index(dst->dtype)](src, dst, dim, reps);
}

}