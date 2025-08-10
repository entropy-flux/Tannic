#include <array>
#include "cpu/concat.hpp"

namespace {

template<typename T>
void stridedConcatKernel(
    const T* srcA_ptr, const shape_t& srcA_shape, const strides_t& srcA_strides,
    const T* srcB_ptr, const shape_t& srcB_shape, const strides_t& srcB_strides,
    T* dst_ptr,
    uint8_t rank,
    int dim
) {
    size_t outer_ne = 1;
    for (int i = 0; i < dim; ++i) { 
        outer_ne *= srcA_shape.sizes[i];
    }

    size_t inner_ne = 1;
    for (int i = dim + 1; i < rank; ++i) { 
        inner_ne *= srcA_shape.sizes[i];
    }

    size_t dimA_size = srcA_shape.sizes[dim];
    size_t dimB_size = srcB_shape.sizes[dim];
    size_t strideA = srcA_strides.sizes[dim];
    size_t strideB = srcB_strides.sizes[dim];

    for (size_t outer_idx = 0; outer_idx < outer_ne; ++outer_idx) {
        size_t outer_offsetA = 0;
        size_t outer_offsetB = 0;

        {
            size_t tmp = outer_idx;
            for (int i = dim - 1; i >= 0; --i) {
                size_t coord = tmp % srcA_shape.sizes[i];
                tmp /= srcA_shape.sizes[i];
                outer_offsetA += coord * srcA_strides.sizes[i];
                outer_offsetB += coord * srcB_strides.sizes[i];
            }
        }

        for (size_t j = 0; j < dimA_size; ++j) {
            const T* src_block = srcA_ptr + outer_offsetA + j * strideA;
            for (size_t k = 0; k < inner_ne; ++k) {
                dst_ptr[k] = src_block[k];
            }
            dst_ptr += inner_ne;
        }
        
        for (size_t j = 0; j < dimB_size; ++j) {
            const T* src_block = srcB_ptr + outer_offsetB + j * strideB;
            for (size_t k = 0; k < inner_ne; ++k) {
                dst_ptr[k] = src_block[k];
            }
            dst_ptr += inner_ne;
        }
    }
}

template<typename T>
status launchConcatKernel(
    const tensor_t* src0, const tensor_t* src1, tensor_t* dst, int axis
) {
    stridedConcatKernel<T>(
        static_cast<const T*>(src0->address), src0->shape, src0->strides,
        static_cast<const T*>(src1->address), src1->shape, src1->strides,
        static_cast<T*>(dst->address), dst->rank, axis
    );
    return SUCCESS;
};

constexpr static status launchDefaultKernel( const tensor_t*, const tensor_t*, tensor_t*, int ) {
    return UNSUPPORTED_DTYPE;
}

constexpr static int index(type dtype) {
    return static_cast<int>(dtype);
}

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, int);

constexpr auto dispatchConcatKernel = []() {
    std::array<Kernel, index(TYPES)> table;
    table.fill(launchDefaultKernel);
    table[index(int8)]     = launchConcatKernel<int8_t>;
    table[index(int16)]    = launchConcatKernel<int16_t>;
    table[index(int32)]    = launchConcatKernel<int32_t>;
    table[index(int64)]    = launchConcatKernel<int64_t>;
    table[index(float32)]  = launchConcatKernel<float>;
    table[index(float64)]  = launchConcatKernel<double>;
    return table;
}();

} namespace cpu {

status concat(const tensor_t* srcA, const tensor_t* srcB, tensor_t* dst, int dim) {
    return dispatchConcatKernel[index(dst->dtype)](srcA, srcB, dst, dim);
}

}