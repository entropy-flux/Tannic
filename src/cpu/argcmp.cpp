#include <stdexcept>
#include <vector>
#include <array>
#include <cmath> 
#include "argcmp.hpp"  

namespace cpu { 

template<typename S, typename D, typename Cmp>
void argCompareKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, uint8_t dim, S initial_value
) {
    Cmp cmp{};

    if (rank == 0) {
        *dst_ptr = 0;
        return;
    }
 
    size_t total = 1;
    for (int i = 0; i < rank; ++i) {
        if (i != dim)
            total *= src_shape.sizes[i];
    } 

    size_t cnt[8] = {0};

    for (size_t idx = 0; idx < total; ++idx) {
        int64_t best_idx = 0;
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
            if (++cnt[d] < src_shape.sizes[d]) {
                break;
            } else {
                cnt[d] = 0;
            }
        }
    }
} 
 
template<typename S, typename Cmp>
status launchArgCompare(const tensor_t* src, tensor_t* dst, uint8_t dim, S init) {    
    argCompareKernel<S, int64_t, Cmp>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<int64_t*>(dst->address), dst->shape, dst->strides,
        src->rank, dim, init
    );
    return SUCCESS;
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

status argmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgCompare<int8_t, GE> (src, dst, dim, std::numeric_limits<int8_t>::lowest());
        case int16:   return launchArgCompare<int16_t, GE>(src, dst, dim, std::numeric_limits<int16_t>::lowest());
        case int32:   return launchArgCompare<int32_t, GE>(src, dst, dim, std::numeric_limits<int32_t>::lowest());
        case int64:   return launchArgCompare<int64_t, GE>(src, dst, dim, std::numeric_limits<int64_t>::lowest());
        case float32: return launchArgCompare<float, GE>  (src, dst, dim, -std::numeric_limits<float>::infinity());
        case float64: return launchArgCompare<double, GE> (src, dst, dim, -std::numeric_limits<double>::infinity());
        default:      return UNSUPPORTED_DTYPE;
    }
}

status argmin(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgCompare<int8_t, LE> (src, dst, dim, std::numeric_limits<int8_t>::max());
        case int16:   return launchArgCompare<int16_t, LE>(src, dst, dim, std::numeric_limits<int16_t>::max());
        case int32:   return launchArgCompare<int32_t, LE>(src, dst, dim, std::numeric_limits<int32_t>::max());
        case int64:   return launchArgCompare<int64_t, LE>(src, dst, dim, std::numeric_limits<int64_t>::max());
        case float32: return launchArgCompare<float, LE>  (src, dst, dim, std::numeric_limits<float>::infinity());
        case float64: return launchArgCompare<double, LE> (src, dst, dim, std::numeric_limits<double>::infinity());
        default:      return UNSUPPORTED_DTYPE;
    }
}


} // namespace cpu