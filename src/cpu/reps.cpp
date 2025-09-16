#include <cstring>   
#include <array>
#include "cpu/reps.hpp"

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

template<typename T>
void stridedRepeatsKernel(
    const T* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    T* dst_ptr, uint8_t rank, int axis, int reps, size_t outer_ne, size_t inner_ne
) { 
    size_t dim_size = src_shape.sizes[axis];
    size_t dim_stride = src_strides.sizes[axis];
    
    for (size_t outer_idx = 0; outer_idx < outer_ne; ++outer_idx) {
        size_t outer_offset = 0;
        size_t tmp = outer_idx;
        for (int i = axis - 1; i >= 0; --i) {
            size_t coord = tmp % src_shape.sizes[i];
            tmp /= src_shape.sizes[i];
            outer_offset += coord * src_strides.sizes[i];
        }

        for (size_t j = 0; j < dim_size; ++j) {
            const T* src_block = src_ptr + outer_offset + j * dim_stride;

            for (int r = 0; r < reps; ++r) {
                size_t idx[8];
                for (size_t k = 0; k < inner_ne; ++k) {
                    size_t tmp_k = k;
                    size_t inner_offset = 0;
                    for (int d = rank - 1; d > axis; --d) {
                        size_t coord = tmp_k % src_shape.sizes[d];
                        tmp_k /= src_shape.sizes[d];
                        inner_offset += coord * src_strides.sizes[d];
                    }
                    dst_ptr[k] = *(src_block + inner_offset);
                }
                dst_ptr += inner_ne;
            }
        }
    }
}

template<typename T>
status launchRepeatsKernel(const tensor_t* src, tensor_t* dst, int axis, int reps) {
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

    stridedRepeatsKernel<T>(
        static_cast<const T*>(src->address), src->shape, src->strides,
        static_cast<T*>(dst->address), dst->rank, axis, reps, outer_ne, inner_ne
    );
    return SUCCESS;
}

}  namespace cpu {

status repeat(const tensor_t* src, tensor_t* dst, int dim, int reps) {
    switch (dst->dtype) {
        case int8:      return launchRepeatsKernel<int8_t>(src, dst, dim, reps);
        case int16:     return launchRepeatsKernel<int16_t>(src, dst, dim, reps);
        case int32:     return launchRepeatsKernel<int32_t>(src, dst, dim, reps);
        case int64:     return launchRepeatsKernel<int64_t>(src, dst, dim, reps);
#if HAS_FLOAT16
        case float16:  return launchRepeatsKernel<half>(src, dst, dim, reps);
        case bfloat16: return launchRepeatsKernel<bhalf>(src, dst, dim, reps);
#endif
        case float32:   return launchRepeatsKernel<float>(src, dst, dim, reps);
        case float64:   return launchRepeatsKernel<double>(src, dst, dim, reps);
        default:        return UNSUPPORTED_DTYPE;
    }
}

}  // namespace cpu
