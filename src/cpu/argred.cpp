#include <array>
#include <cmath> 
#include <limits>
#include <cstdint> 
#include "argred.hpp"
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
    
template<typename S, typename D, typename Op>
void contiguousArgReduceKernel(
    const S* src_ptr, const shape_t& src_shape,
    D* dst_ptr,
    uint8_t rank, uint8_t ax, size_t ne,
    size_t outer_sizes, size_t inner_sizes
) {
    Op op; 
    size_t reduce_dim = src_shape.sizes[ax];
    
    for (size_t outer = 0; outer < outer_sizes; ++outer) {
        for (size_t inner = 0; inner < inner_sizes; ++inner) {
            const S* slice_start = src_ptr + (outer * reduce_dim * inner_sizes + inner);
            D accum = D(0); 
            for (size_t i = 0; i < reduce_dim; ++i) {
                const S val = slice_start[i * inner_sizes];
                accum = op(accum, val);
            }
            
            dst_ptr[outer * inner_sizes + inner] = op.finalize(accum, reduce_dim);
        }
    }
} 

template<typename S, typename D, typename Op>
void stridedArgReduceKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr,
    uint8_t rank, uint8_t ax, size_t ne
) {
    Op op;  
    size_t cnt[8] = {0};

    for (size_t idx = 0; idx < ne; ++idx) {
        D accum = D(0);

        for (size_t i = 0; i < src_shape.sizes[ax]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == ax) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }

            const S val = src_ptr[offset];
            accum = op(accum, val);
        }

        dst_ptr[idx] = op.finalize(accum, src_shape.sizes[ax]);
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (dim == ax) continue; 
            if (++cnt[dim] < src_shape.sizes[dim]) {
                break;
            } else {
                cnt[dim] = 0;
            }
        }
    }
}
 

template<typename S, typename D, typename Op>
void contiguousArgCompareKernel(
    const S* src_ptr, const shape_t& src_shape,
    D* dst_ptr,
    uint8_t rank, uint8_t ax, size_t ne, S initial_value,
    size_t outer_sizes, size_t inner_sizes
) {
    Op cmp{};  
    size_t reduce_dim = src_shape.sizes[ax]; 
    
    for (size_t outer = 0; outer < outer_sizes; ++outer) {
        for (size_t inner = 0; inner < inner_sizes; ++inner) {
            const S* slice_start = src_ptr + (outer * reduce_dim * inner_sizes + inner);
            
            int64_t best_idx = 0;
            S best_val = initial_value;
             
            for (size_t i = 0; i < reduce_dim; ++i) {
                const S val = slice_start[i * inner_sizes];
                if (cmp(val, best_val) || (val == best_val && i < best_idx)) {
                    best_val = val;
                    best_idx = i;
                }
            }
            
            dst_ptr[outer * inner_sizes + inner] = static_cast<D>(best_idx);
        }
    }
}

template<typename S, typename D, typename Op>
void stridedArgCompareKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr,
    uint8_t rank, uint8_t ax, size_t ne, S initial_value
) {
    Op cmp{}; 
    size_t cnt[8] = {0};

    for (size_t idx = 0; idx < ne; ++idx) {
        int64_t best_idx = 0;
        S best_val = initial_value;

        for (size_t i = 0; i < src_shape.sizes[ax]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == ax) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }

            const S val = src_ptr[offset];
            if (cmp(val, best_val) || (val == best_val && i < best_idx)) {
                best_val = val;
                best_idx = i;
            }
        }

        *dst_ptr++ = static_cast<D>(best_idx); 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (dim == ax) continue; 
            if (++cnt[dim] < src_shape.sizes[dim]) {
                break;
            } else {
                cnt[dim] = 0;
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

template<typename S, typename D, typename Op>
status launchArgReduce(const tensor_t* src, tensor_t* dst, uint8_t ax) { 
    if (src->layout == CONTIGUOUS) {  
        size_t ne = 1;
        shape_t src_shape; 
        size_t outer_size = 1;
        size_t inner_size = 1; 
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            if (dim < ax) { 
                src_shape.sizes[dim] = src->shape.sizes[dim];
                ne *= src->shape.sizes[dim];
                outer_size *= src->shape.sizes[dim];
            } 

            else if(dim == ax) {
                src_shape.sizes[dim] = src->shape.sizes[dim]; 
            }
            
            else { 
                src_shape.sizes[dim] = src->shape.sizes[dim];
                ne *= src->shape.sizes[dim];
                inner_size *= src->shape.sizes[dim]; 
            }
        } 

        contiguousArgReduceKernel<S, D, Op>(
            reinterpret_cast<const S*>(src->address), src_shape,
            reinterpret_cast<D*>(dst->address),
            src->rank, ax, ne, outer_size, inner_size
        );
        return SUCCESS;
    } 
    
    else {     
        size_t ne = 1;
        shape_t src_shape;
        strides_t src_strides;
        for (int dim = 0; dim < src->rank; ++dim) {
            src_shape.sizes[dim] = src->shape.sizes[dim];
            src_strides.sizes[dim] = src->strides.sizes[dim];
            if (dim != ax)
                ne *= src_shape.sizes[dim];
        } 

        stridedArgReduceKernel<S, D, Op>(
            reinterpret_cast<const S*>(src->address), src_shape, src_strides,
            reinterpret_cast<D*>(dst->address),
            src->rank, ax, ne
        );
        return SUCCESS;
    } 
} 
 
template<typename S, typename Op>
status launchArgCompare(const tensor_t* src, tensor_t* dst, uint8_t ax, S init) { 
    if (src->layout == CONTIGUOUS) {  
        size_t ne = 1;
        shape_t src_shape; 
        size_t outer_size = 1;
        size_t inner_size = 1; 
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            if (dim < ax) { 
                src_shape.sizes[dim] = src->shape.sizes[dim];
                ne *= src->shape.sizes[dim];
                outer_size *= src->shape.sizes[dim];
            } 

            else if(dim == ax) {
                src_shape.sizes[dim] = src->shape.sizes[dim]; 
            }
            
            else { 
                src_shape.sizes[dim] = src->shape.sizes[dim];
                ne *= src->shape.sizes[dim];
                inner_size *= src->shape.sizes[dim]; 
            }
        } 
        contiguousArgCompareKernel<S, int64_t, Op>(
            reinterpret_cast<const S*>(src->address), src_shape,
            reinterpret_cast<int64_t*>(dst->address),
            src->rank, ax, ne, init,
            outer_size, inner_size
        );
        return SUCCESS;

    } else {     
        size_t ne = 1;
        shape_t src_shape;
        strides_t src_strides;
        for (int dim = 0; dim < src->rank; ++dim) {
            src_shape.sizes[dim] = src->shape.sizes[dim];
            src_strides.sizes[dim] = src->strides.sizes[dim];
            if (dim != ax) ne *= src_shape.sizes[dim];
        } 

        stridedArgCompareKernel<S, int64_t, Op>(
            reinterpret_cast<const S*>(src->address), src_shape, src_strides,
            reinterpret_cast<int64_t*>(dst->address),
            src->rank, ax, ne, init
        );
        return SUCCESS;
    }
} 

} namespace cpu {

status argsum(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgReduce<int8_t, int8_t, Sum>(src, dst, dim);
        case int16:   return launchArgReduce<int16_t, int16_t, Sum>(src, dst, dim);
        case int32:   return launchArgReduce<int32_t, int32_t, Sum>(src, dst, dim);
        case int64:   return launchArgReduce<int64_t, int64_t, Sum>(src, dst, dim);
        case float32: return launchArgReduce<float, float,   Sum>(src, dst, dim);
        case float64: return launchArgReduce<double, double, Sum>(src, dst, dim);
#if HAS_FLOAT16
        case float16:  return launchArgReduce<half, half,     Sum>(src, dst, dim);
        case bfloat16: return launchArgReduce<bhalf, bhalf,     Sum>(src, dst, dim);
#endif
        default:      return UNSUPPORTED_DTYPE;
    }
}

status argmean(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case float32: return launchArgReduce<float, float,  Mean>(src, dst, dim);
        case float64: return launchArgReduce<double, double, Mean>(src, dst, dim);
#if HAS_FLOAT16
        case float16:  return launchArgReduce<half, half,     Mean>(src, dst, dim);
        case bfloat16: return launchArgReduce<bhalf, bhalf,     Mean>(src, dst, dim);
#endif
        default:      return UNSUPPORTED_DTYPE;
    }
} 

status argmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case int8:    return launchArgCompare<int8_t, GE> (src, dst, dim, std::numeric_limits<int8_t>::lowest());
        case int16:   return launchArgCompare<int16_t, GE>(src, dst, dim, std::numeric_limits<int16_t>::lowest());
        case int32:   return launchArgCompare<int32_t, GE>(src, dst, dim, std::numeric_limits<int32_t>::lowest());
        case int64:   return launchArgCompare<int64_t, GE>(src, dst, dim, std::numeric_limits<int64_t>::lowest());
        case float32: return launchArgCompare<float, GE>  (src, dst, dim, -std::numeric_limits<float>::infinity());
        case float64: return launchArgCompare<double, GE> (src, dst, dim, -std::numeric_limits<double>::infinity());
#if HAS_FLOAT16
        case float16: return launchArgCompare<half, GE>   (src, dst, dim, half(-std::numeric_limits<float>::infinity()));
        case bfloat16:return launchArgCompare<bhalf, GE>  (src, dst, dim, bhalf(-std::numeric_limits<float>::infinity()));
#endif
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
#if HAS_FLOAT16
        case float16: return launchArgCompare<half, LE>   (src, dst, dim, half(std::numeric_limits<float>::infinity()));
        case bfloat16:return launchArgCompare<bhalf, LE>  (src, dst, dim, bhalf(std::numeric_limits<float>::infinity()));
#endif
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu