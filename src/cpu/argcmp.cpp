#include <stdexcept>
#include <vector>
#include <array>
#include <cmath> 
#include "argcmp.hpp"  

template<typename S, typename D, typename Cmp>
void argCompareKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, uint8_t dim, const void* init_val
) { 
    Cmp cmp{}; 
    S initial_value = *static_cast<const S*>(init_val);

    if (rank == 0) {
        *dst_ptr = 0;
        return;
    }

    size_t total = 1;
    for (int i = 0; i < rank; ++i)
        if (i != dim) total *= dst_shape.sizes[i];


    size_t cnt[8] = {0};
    for (size_t idx = 0; idx < total; ++idx) {
        size_t best_idx = 0;
        S best_val = initial_value;

        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }

            const S val = src_ptr[offset];
            if (cmp(val, best_val) || (val == best_val && i < best_idx)) {
                best_val = val;
                best_idx = i;
            }
        }

        *dst_ptr++ = static_cast<D>(best_idx);
 
        for (int d = rank - 1; d >= 0; --d) {
            if (d == dim) continue;
            if (++cnt[d] < dst_shape.sizes[d]) break;
            cnt[d] = 0;
        }
    }
}


struct GE {
    template<class A, class B>
    bool operator()(A&& a, B&& b) const noexcept {
        return a > b;
    }
};

struct LE { 
    template<class A, class B>
    bool operator()(A&& a, B&& b) const noexcept {
        return a < b;
    }
};
 

#include <iostream>

namespace cpu { 

status argmax(tensor_t const* src, tensor_t* dst, uint8_t dim) {   
    switch (src->dtype) { 
        case int8: {
            static const int8_t init = std::numeric_limits<int8_t>::lowest();
            argCompareKernel<int8_t, int64_t, GE>(
                (int8_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int16: {
            static const int16_t init = std::numeric_limits<int16_t>::lowest();
            argCompareKernel<int16_t, int64_t, GE>(
                (int16_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int32: {
            static const int32_t init = std::numeric_limits<int32_t>::lowest();
            argCompareKernel<int32_t, int64_t, GE>(
                (int32_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int64: {
            static const int64_t init = std::numeric_limits<int64_t>::lowest();
            argCompareKernel<int64_t, int64_t, GE>(
                (int64_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case float32: {
            static const float init = -std::numeric_limits<float>::infinity();
            argCompareKernel<float, int64_t, GE>(
                (float*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case float64: {
            static const double init = -std::numeric_limits<double>::infinity();
            argCompareKernel<double, int64_t, GE>(
                (double*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        default:
            return UNSUPORTED_DTYPE;
    }  
}

status argmin(tensor_t const* src, tensor_t* dst, uint8_t dim) {   
    switch (src->dtype) { 
        case int8: {
            static const int8_t init = std::numeric_limits<int8_t>::max();
            argCompareKernel<int8_t, int64_t, LE>(
                (int8_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int16: {
            static const int16_t init = std::numeric_limits<int16_t>::max();
            argCompareKernel<int16_t, int64_t, LE>(
                (int16_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int32: {
            static const int32_t init = std::numeric_limits<int32_t>::max();
            argCompareKernel<int32_t, int64_t, LE>(
                (int32_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case int64: {
            static const int64_t init = std::numeric_limits<int64_t>::max();
            argCompareKernel<int64_t, int64_t, LE>(
                (int64_t*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case float32: {
            static const float init = std::numeric_limits<float>::infinity();
            argCompareKernel<float, int64_t, LE>(
                (float*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        case float64: {
            static const double init = std::numeric_limits<double>::infinity();
            argCompareKernel<double, int64_t, LE>(
                (double*)(src->address), src->shape, src->strides,
                (int64_t*)(dst->address), dst->shape, dst->strides,
                src->rank, dim, &init);
            return SUCCESS;
        }
        default:
            return UNSUPORTED_DTYPE;
    }
}

} // namespace cpu