#include <cmath>
#include <array>
#include "cpu/ops.hpp"
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

template<typename S0, typename S1, typename D, class Op>
void contiguousScaleKernel(
    const S0* src0_ptr,
    S1 scalar_value,
    D* dst_ptr, 
    size_t ne
) {
    Op op{};
    for (size_t i = 0; i < ne; ++i) {
        dst_ptr[i] = op(src0_ptr[i], scalar_value);
    }
} 

template<typename S0, typename S1, typename D, class Op>
void stridedScaleOpKernel(
    const S0* src0_ptr, const shape_t& src0_shape, const strides_t& src0_strides,
    S1 scalar_value,
    D* dst_ptr,
    uint8_t rank, size_t ne
) {
    Op op{};
    size_t cnt[8] = {0};

    for (size_t idx = 0; idx < ne; ++idx) {
        size_t offs0 = 0;
 
        for (int dim = 0; dim < rank; ++dim) {
            size_t coord = (src0_shape.sizes[dim] == 1) ? 0 : cnt[dim];
            offs0 += coord * src0_strides.sizes[dim];
        }

        dst_ptr[idx] = static_cast<D>(op(src0_ptr[offs0], scalar_value));
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < src0_shape.sizes[dim])
                break;
            cnt[dim] = 0;
        }
    }
} 


template<typename S0, typename S1, typename D, class Op>
status launchScaleOpKernel(const tensor_t* src0, const scalar_t* src1, tensor_t* dst) {
    const size_t ne = dst->size;   
    S1 scalar = *reinterpret_cast<const S1*>(src1->address); 
    if(src0->layout == CONTIGUOUS) { 
        contiguousScaleKernel<S0, S1, D, Op> (
            reinterpret_cast<const S0*>(src0->address), scalar,
            reinterpret_cast<D*>(dst->address), ne
        );
        return SUCCESS;

    } else { 
        shape_t src_shape; 
        strides_t src_strides; 
        for (int dim = 0; dim < src0->rank; ++dim) {
            src_shape.sizes[dim] = dst->shape.sizes[dim];
            src_strides.sizes[dim] = src0->strides.sizes[dim];
        } 
        stridedScaleOpKernel<S0, S1, D, Op> (
            reinterpret_cast<const S0*>(src0->address), src_shape, src_strides,
            scalar,
            reinterpret_cast<D*>(dst->address), dst->rank, ne
        );
        return SUCCESS;
    } 
}

struct Mul {
    template<class A, class B>
    inline auto operator()(A a, B b) const noexcept(noexcept(a * b)) {
        return a * b;
    }
};

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  
 
constexpr static status launchDefaultBinaryOpKernel(const tensor_t* src0, const scalar_t* src1, tensor_t* dst) {
    return UNSUPPORTED_DTYPE;
};     

using Kernel = status(*)(const tensor_t*, const scalar_t*, tensor_t*);

constexpr auto dispatchMul = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel); 
    table[index(int8 , int8)]  = launchScaleOpKernel<uint8_t, uint8_t, uint8_t, Mul>;
    table[index(int16, int16)] = launchScaleOpKernel<uint16_t, uint16_t, uint16_t, Mul>;
    table[index(int32, int32)] = launchScaleOpKernel<uint32_t, uint32_t, uint32_t, Mul>;
    table[index(int64, int64)] = launchScaleOpKernel<uint64_t, uint64_t, uint64_t, Mul>;
#if HAS_FLOAT16
    table[index(float16, float16)] = launchScaleOpKernel<half, half, half, Mul>;
    table[index(bfloat16, bfloat16)] = launchScaleOpKernel<bhalf, bhalf, bhalf, Mul>;
#endif  
    table[index(float32, float32)] = launchScaleOpKernel<float, float, float, Mul>;
    table[index(float64, float64)] = launchScaleOpKernel<double, double, double, Mul>; 
    return table;
}();

} namespace cpu {

status scale(const tensor_t* src0, const scalar_t* src1, tensor_t* dst) {
    return dispatchMul[index(src0->dtype, src1->dtype)](src0, src1, dst);
}

}