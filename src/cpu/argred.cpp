#include "argred.hpp"

namespace { 
    
template<typename S, typename D, typename Op>
void argReduceKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, uint8_t dim
) {
    Op op;
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
        D accum = D(0);

        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }

            const S val = src_ptr[offset];
            accum = op(accum, val);
        }

        dst_ptr[idx] = op.finalize(accum, src_shape.sizes[dim]);
 
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
 
struct Sum {
    template<typename A, typename B>
    A operator()(A accum, B val) const { return accum + val; }
    
    template<typename T>
    T finalize(T accum, size_t count) const { return accum; }
};
 
struct Mean {  
    template<typename A, typename B>
    A operator()(A accum, B val) const { return accum + val; }
    
    template<typename T>
    T finalize(T accum, size_t count) const { 
        return static_cast<T>(accum) / static_cast<T>(count); 
    }
};

template<typename S, typename D, typename Op>
status launchArgReduce(const tensor_t* src, tensor_t* dst, uint8_t dim) {    
    argReduceKernel<S, D, Op>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<D*>(dst->address), dst->shape, dst->strides,
        src->rank, dim
    );
    return SUCCESS;
} 

} namespace cpu {

status argsum(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgReduce<int8_t, int8_t, Sum>(src, dst, dim);
        case int16:   return launchArgReduce<int16_t, int16_t, Sum>(src, dst, dim);
        case int32:   return launchArgReduce<int32_t, int32_t, Sum>(src, dst, dim);
        case int64:   return launchArgReduce<int64_t, int64_t, Sum>(src, dst, dim);
        case float32: return launchArgReduce<float, float,   Sum>(src, dst, dim);
        case float64: return launchArgReduce<double,  double, Sum>(src, dst, dim);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status argmean(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case float32: return launchArgReduce<float, float,  Mean>(src, dst, dim);
        case float64: return launchArgReduce<double, double,  Mean>(src, dst, dim);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu