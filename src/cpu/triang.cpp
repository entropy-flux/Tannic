#include "triang.hpp"

namespace {

struct Upper {
    int k;
    Upper(int diag = 0) : k(diag) {}

    template<typename T>
    T operator()(T value, size_t i, size_t j) const noexcept {
        return (j >= i + k) ? value : T(0);
    }
};

struct Lower {
    int k; 
    Lower(int diag = 0) : k(diag) {}

    template<typename T>
    T operator()(T value, size_t i, size_t j) const noexcept {
        return (j <= i + k) ? value : T(0);
    }
};

template<typename T, class Fn>
void triangularKernel(const T* src_ptr, T* dst_ptr, const shape_t& shape, Fn tri) {
    size_t rows = shape.sizes[0];
    size_t cols = shape.sizes[1];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst_ptr[i * cols + j] = tri(src_ptr[i * cols + j], i, j);
        }
    }
}

template<typename T, class Fn>
void stridedTriangularKernel(
    const T* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    T* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    Fn tri
) {
    size_t rows = src_shape.sizes[0];
    size_t cols = src_shape.sizes[1];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t src_offset = i * src_strides.sizes[0] + j * src_strides.sizes[1];
            size_t dst_offset = i * dst_strides.sizes[0] + j * dst_strides.sizes[1];
            dst_ptr[dst_offset] = tri(src_ptr[src_offset], i, j);
        }
    }
}

template<typename T, class Fn>
status launchTriangularKernel(const tensor_t* src, tensor_t* dst, Fn tri) {
    if (src->rank != 2) return UNSUPPORTED_DTYPE;

    if (src->layout == CONTIGUOUS) {
        triangularKernel<T>((const T*)src->address, (T*)dst->address, dst->shape, tri);
        return SUCCESS;
    } else {
        stridedTriangularKernel<T>(
            (const T*)src->address, src->shape, src->strides,
            (T*)dst->address, dst->shape, dst->strides, tri
        );
        return SUCCESS;
    }
}

} // namespace

namespace cpu {

status triu(const tensor_t* src, tensor_t* dst, int k) {
    Upper tri(k);
    switch (src->dtype) {
        case int32:      return launchTriangularKernel<int32_t>(src, dst, tri);
        case int64:      return launchTriangularKernel<int64_t>(src, dst, tri);
        case float32:    return launchTriangularKernel<float>(src, dst, tri);
        case float64:    return launchTriangularKernel<double>(src, dst, tri);
#if HAS_FLOAT16
        case float16:    return launchTriangularKernel<half>(src, dst, tri);
        case bfloat16:   return launchTriangularKernel<bhalf>(src, dst, tri);
#endif
        default:         return UNSUPPORTED_DTYPE;
    }
}

status tril(const tensor_t* src, tensor_t* dst, int k) {
    Lower tri(k);
    switch (src->dtype) {
        case int32:      return launchTriangularKernel<int32_t>(src, dst, tri);
        case int64:      return launchTriangularKernel<int64_t>(src, dst, tri);
        case float32:    return launchTriangularKernel<float>(src, dst, tri);
        case float64:    return launchTriangularKernel<double>(src, dst, tri);
#if HAS_FLOAT16
        case float16:    return launchTriangularKernel<half>(src, dst, tri);
        case bfloat16:   return launchTriangularKernel<bhalf>(src, dst, tri);
#endif
        default:         return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu
