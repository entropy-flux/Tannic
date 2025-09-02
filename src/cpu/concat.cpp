#include <array>
#include "cpu/concat.hpp"
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
void stridedConcatKernel(
    const T* srcA_ptr, const shape_t& srcA_shape, const strides_t& srcA_strides,
    const T* srcB_ptr, const shape_t& srcB_shape, const strides_t& srcB_strides,
    T* dst_ptr,
    uint8_t rank,
    int dim
) {
    // Calculate total elements in each dimension section
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

    // For each outer slice
    for (size_t outer_idx = 0; outer_idx < outer_ne; ++outer_idx) {
        // Calculate the coordinate in the outer dimensions
        size_t coords[8];
        size_t temp = outer_idx;
        for (int i = dim - 1; i >= 0; --i) {
            coords[i] = temp % srcA_shape.sizes[i];
            temp /= srcA_shape.sizes[i];
        }

        // Calculate offsets for source tensors
        size_t offsetA = 0;
        size_t offsetB = 0;
        for (int i = 0; i < dim; ++i) {
            offsetA += coords[i] * srcA_strides.sizes[i];
            offsetB += coords[i] * srcB_strides.sizes[i];
        }

        // Calculate destination offset for this outer slice
        size_t dst_offset = outer_idx * (dimA_size + dimB_size) * inner_ne;

        // Copy from A
        for (size_t j = 0; j < dimA_size; ++j) {
            size_t src_offset_A = offsetA + j * srcA_strides.sizes[dim];
            size_t dst_offset_A = dst_offset + j * inner_ne;
            
            for (size_t k = 0; k < inner_ne; ++k) {
                size_t inner_offset = (dim == rank - 1) ? k : k * srcA_strides.sizes[dim + 1];
                dst_ptr[dst_offset_A + k] = srcA_ptr[src_offset_A + inner_offset];
            }
        }

        // Copy from B
        for (size_t j = 0; j < dimB_size; ++j) {
            size_t src_offset_B = offsetB + j * srcB_strides.sizes[dim];
            size_t dst_offset_B = dst_offset + (dimA_size + j) * inner_ne;
            
            for (size_t k = 0; k < inner_ne; ++k) {
                size_t inner_offset = (dim == rank - 1) ? k : k * srcB_strides.sizes[dim + 1];
                dst_ptr[dst_offset_B + k] = srcB_ptr[src_offset_B + inner_offset];
            }
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
    table[index(float16)]  = launchConcatKernel<half>;
    table[index(float32)]  = launchConcatKernel<float>;
    table[index(float64)]  = launchConcatKernel<double>;
    return table;
}();

} // namespace

namespace cpu {

status concat(const tensor_t* srcA, const tensor_t* srcB, tensor_t* dst, int dim) {
    return dispatchConcatKernel[index(dst->dtype)](srcA, srcB, dst, dim);
}

}