#include <stdexcept>
#include <vector>
#include <array>
#include <complex>
#include <stdfloat>
#include "cpu/ops.hpp" 

namespace {   

template<typename S0, typename S1, typename D, class Op>
void singletonBinaryOpKernel(
    const S0* src0_ptr,
    const S1* src1_ptr,
    D* dst_ptr
) {
    Op op{};
    *dst_ptr = op(*src0_ptr, *src1_ptr);
}

template<typename S0, typename S1, typename D, class Op>
void contiguousBinaryOpKernel(const S0* src0_ptr, const S1* src1_ptr, D* dst_ptr, size_t ne) {
    Op op{};
    for (size_t i = 0; i < ne; ++i) {
        dst_ptr[i] = op(src0_ptr[i], src1_ptr[i]);
    }
}

template<typename S0, typename S1, typename D, class Op>
void stridedBinaryOpKernel(
    const S0* src0_ptr, strides_t src0_strides,
    const S1* src1_ptr, strides_t src1_strides,
    D* dst_ptr, const strides_t& dst_strides,
    const shape_t& dst_shape, uint8_t dst_rank,
    size_t ne
) {
    Op op{};
    int rank = static_cast<int>(dst_rank);
    for (size_t linear_idx = 0; linear_idx < ne; ++linear_idx) {
        size_t offset0 = 0, offset1 = 0, offset_dst = 0;
        size_t remaining = linear_idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            const size_t coord = remaining % dst_shape.sizes[dim];
            remaining /= dst_shape.sizes[dim]; 
            offset0    += coord * static_cast<size_t>(src0_strides.sizes[dim]);
            offset1    += coord * static_cast<size_t>(src1_strides.sizes[dim]);
            offset_dst += coord * static_cast<size_t>(dst_strides.sizes[dim]);
        }

        dst_ptr[offset_dst] = op(src0_ptr[offset0], src1_ptr[offset1]);
    }
}
 

template<typename S0, typename S1, typename D, class Op>
status launchBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    const size_t ne = dst->size; 
    if (dst->layout == SINGLETON) {
        singletonBinaryOpKernel<S0, S1, D, Op>(
            static_cast<const S0*>(src0->address),
            static_cast<const S1*>(src1->address),
            static_cast<D*>(dst->address)
        );
        return SUCCESS;
    }
 
    else { 
        strides_t src0_strides{};
        strides_t src1_strides{};

        bool flat = true;               
        int64_t expect_src0 = 1;       
        int64_t expect_src1 = 1;        

        const int rank = dst->rank;
        const int off0 = rank - src0->rank; 
        const int off1 = rank - src1->rank;

        for (int dim = rank - 1; dim >= 0; --dim) {
            const size_t sz   = dst->shape.sizes[dim];
            const size_t sz0 = (dim >= off0) ? src0->shape.sizes[dim - off0] : 1;
            const size_t sz1 = (dim >= off1) ? src1->shape.sizes[dim - off1] : 1;
    
            src0_strides.sizes[dim] = (sz0 == 1) ? 0 : src0->strides.sizes[dim - off0];
            src1_strides.sizes[dim] = (sz1 == 1) ? 0 : src1->strides.sizes[dim - off1];
    
            if (!(sz0 == sz && ((dim >= off0) ? (src0->strides.sizes[dim - off0] == expect_src0) : (expect_src0 == 1))))
                flat = false;

            if (!(sz1 == sz && ((dim >= off1) ? (src1->strides.sizes[dim - off1] == expect_src1) : (expect_src1 == 1))))
                flat = false;
    
            expect_src0 *= (sz0 == 1 ? 1 : static_cast<int64_t>(sz0));
            expect_src1 *= (sz1 == 1 ? 1 : static_cast<int64_t>(sz1));
        }

        if (flat) {
            contiguousBinaryOpKernel<S0, S1, D, Op>(
                static_cast<const S0*>(src0->address),
                static_cast<const S1*>(src1->address),
                static_cast<D*>(dst->address),
                ne
            );
        } 
        
        else {
            stridedBinaryOpKernel<S0, S1, D, Op>(
                static_cast<const S0*>(src0->address), src0_strides,
                static_cast<const S1*>(src1->address), src1_strides,
                static_cast<D*>(dst->address), dst->strides,
                dst->shape, dst->rank,
                ne
            );
        }

        return SUCCESS;
    } 
}
 
struct Add {
    template<class A, class B>
    inline auto operator()(A a, B b) const noexcept(noexcept(a + b)) {
        return a + b;
    }

    inline std::float16_t operator()(std::float16_t a, std::float16_t b) const noexcept {
        return std::float16_t(float(a) + float(b));
    }

    inline float operator()(std::float16_t a, float b) const noexcept {
        return float(a) + b;
    }

    inline double operator()(std::float16_t a, double b) const noexcept {
        return static_cast<double>(a) + b;
    }

    inline float operator()(float a, std::float16_t b) const noexcept {
        return a + float(b);
    }

    inline double operator()(double a, std::float16_t b) const noexcept {
        return a + static_cast<double>(b);
    }

    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, std::float16_t> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    inline float operator()(std::float16_t a, B b) const noexcept {
        return float(a) + static_cast<float>(b);
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, std::float16_t> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    inline float operator()(A a, std::float16_t b) const noexcept {
        return static_cast<float>(a) + float(b);
    }
};
 
struct Sub {
    template<class A, class B>
    inline auto operator()(A a, B b) const noexcept(noexcept(a - b)) {
        return a - b;
    }

    inline std::float16_t operator()(std::float16_t a, std::float16_t b) const noexcept {
        return std::float16_t(float(a) - float(b));
    }

    inline float operator()(std::float16_t a, float b) const noexcept {
        return float(a) - b;
    }

    inline double operator()(std::float16_t a, double b) const noexcept {
        return static_cast<double>(a) - b;
    }

    inline float operator()(float a, std::float16_t b) const noexcept {
        return a - float(b);
    }

    inline double operator()(double a, std::float16_t b) const noexcept {
        return a - static_cast<double>(b);
    }

    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, std::float16_t> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    inline float operator()(std::float16_t a, B b) const noexcept {
        return float(a) - static_cast<float>(b);
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, std::float16_t> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    inline float operator()(A a, std::float16_t b) const noexcept {
        return static_cast<float>(a) - float(b);
    }
};
 
struct Mul {
    template<class A, class B>
    inline auto operator()(A a, B b) const noexcept(noexcept(a * b)) {
        return a * b;
    }

    inline std::float16_t operator()(std::float16_t a, std::float16_t b) const noexcept {
        return std::float16_t(float(a) * float(b));
    }

    inline float operator()(std::float16_t a, float b) const noexcept {
        return float(a) * b;
    }

    inline double operator()(std::float16_t a, double b) const noexcept {
        return static_cast<double>(a) * b;
    }

    inline float operator()(float a, std::float16_t b) const noexcept {
        return a * float(b);
    }

    inline double operator()(double a, std::float16_t b) const noexcept {
        return a * static_cast<double>(b);
    }

    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, std::float16_t> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    inline float operator()(std::float16_t a, B b) const noexcept {
        return float(a) * static_cast<float>(b);
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, std::float16_t> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    inline float operator()(A a, std::float16_t b) const noexcept {
        return static_cast<float>(a) * float(b);
    }
};
 
struct Pow {
    template<class A, class B>
    inline auto operator()(A a, B b) const noexcept(noexcept(std::pow(a, b))) {
        return std::pow(a, b);
    }

    inline std::float16_t operator()(std::float16_t a, std::float16_t b) const noexcept {
        return std::float16_t(std::pow(float(a), float(b)));
    }

    inline float operator()(std::float16_t a, float b) const noexcept {
        return std::pow(float(a), b);
    }

    inline double operator()(std::float16_t a, double b) const noexcept {
        return std::pow(static_cast<double>(a), b);
    }

    inline float operator()(float a, std::float16_t b) const noexcept {
        return std::pow(a, float(b));
    }

    inline double operator()(double a, std::float16_t b) const noexcept {
        return std::pow(a, static_cast<double>(b));
    }

    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, std::float16_t> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    inline float operator()(std::float16_t a, B b) const noexcept {
        return std::pow(float(a), static_cast<float>(b));
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, std::float16_t> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    inline float operator()(A a, std::float16_t b) const noexcept {
        return std::pow(static_cast<float>(a), float(b));
    }
};

  
using BinaryOpKernel = status(*)( const tensor_t*, const tensor_t*, tensor_t*);      

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  
 
constexpr static status launchDefaultBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    return UNSUPPORTED_DTYPE;
};     

constexpr auto dispatchAdd = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultBinaryOpKernel);

    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Add>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Add>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Add>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Add>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Add>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Add>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Add>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Add>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Add>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Add>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Add>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Add>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Add>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Add>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Add>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Add>;
 
    table[index(float16, float16)] = launchBinaryOpKernel<std::float16_t, std::float16_t, std::float16_t, Add>;
    table[index(float16, float32)] = launchBinaryOpKernel<std::float16_t, float, float, Add>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, std::float16_t, float, Add>;
    table[index(float16, float64)] = launchBinaryOpKernel<std::float16_t, double, double, Add>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, std::float16_t, double, Add>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<std::float16_t, int8_t, float, Add>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, std::float16_t, float, Add>;
    table[index(float16, int16)]   = launchBinaryOpKernel<std::float16_t, int16_t, float, Add>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, std::float16_t, float, Add>;
    table[index(float16, int32)]   = launchBinaryOpKernel<std::float16_t, int32_t, float, Add>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, std::float16_t, float, Add>;
    table[index(float16, int64)]   = launchBinaryOpKernel<std::float16_t, int64_t, double, Add>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, std::float16_t, double, Add>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Add>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Add>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Add>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Add>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Add>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Add>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Add>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Add>;
    
    table[index(complex64, complex64)]   = launchBinaryOpKernel<std::complex<float>,  std::complex<float>,  std::complex<float>,  Add>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<std::complex<double>, std::complex<double>, std::complex<double>, Add>;
    return table;
}();
 
constexpr auto dispatchSub = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultBinaryOpKernel);

    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Sub>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Sub>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Sub>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Sub>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Sub>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Sub>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Sub>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Sub>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Sub>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Sub>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Sub>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Sub>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Sub>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Sub>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Sub>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Sub>;

    table[index(float16, float16)] = launchBinaryOpKernel<std::float16_t, std::float16_t, std::float16_t, Sub>;
    table[index(float16, float32)] = launchBinaryOpKernel<std::float16_t, float, float, Sub>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, std::float16_t, float, Sub>;
    table[index(float16, float64)] = launchBinaryOpKernel<std::float16_t, double, double, Sub>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, std::float16_t, double, Sub>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<std::float16_t, int8_t, float, Sub>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, std::float16_t, float, Sub>;
    table[index(float16, int16)]   = launchBinaryOpKernel<std::float16_t, int16_t, float, Sub>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, std::float16_t, float, Sub>;
    table[index(float16, int32)]   = launchBinaryOpKernel<std::float16_t, int32_t, float, Sub>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, std::float16_t, float, Sub>;
    table[index(float16, int64)]   = launchBinaryOpKernel<std::float16_t, int64_t, double, Sub>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, std::float16_t, double, Sub>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Sub>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Sub>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Sub>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Sub>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Sub>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Sub>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Sub>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Sub>;

    table[index(complex64, complex64)]   = launchBinaryOpKernel<std::complex<float>,  std::complex<float>,  std::complex<float>,  Sub>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<std::complex<double>, std::complex<double>, std::complex<double>, Sub>;
    return table;
}();
 
constexpr auto dispatchMul = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultBinaryOpKernel);

    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Mul>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Mul>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Mul>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Mul>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Mul>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Mul>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Mul>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Mul>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Mul>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Mul>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Mul>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Mul>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Mul>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Mul>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Mul>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Mul>;

    // std::float16_t variants
    table[index(float16, float16)] = launchBinaryOpKernel<std::float16_t, std::float16_t, std::float16_t, Mul>;
    table[index(float16, float32)] = launchBinaryOpKernel<std::float16_t, float, float, Mul>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, std::float16_t, float, Mul>;
    table[index(float16, float64)] = launchBinaryOpKernel<std::float16_t, double, double, Mul>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, std::float16_t, double, Mul>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<std::float16_t, int8_t, float, Mul>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, std::float16_t, float, Mul>;
    table[index(float16, int16)]   = launchBinaryOpKernel<std::float16_t, int16_t, float, Mul>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, std::float16_t, float, Mul>;
    table[index(float16, int32)]   = launchBinaryOpKernel<std::float16_t, int32_t, float, Mul>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, std::float16_t, float, Mul>;
    table[index(float16, int64)]   = launchBinaryOpKernel<std::float16_t, int64_t, double, Mul>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, std::float16_t, double, Mul>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Mul>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Mul>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Mul>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Mul>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Mul>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Mul>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Mul>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Mul>;

    table[index(complex64, complex64)]   = launchBinaryOpKernel<std::complex<float>,  std::complex<float>,  std::complex<float>,  Mul>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<std::complex<double>, std::complex<double>, std::complex<double>, Mul>;
    return table;
}();
 
constexpr auto dispatchPow = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultBinaryOpKernel); 
 
    table[index(float16, float16)] = launchBinaryOpKernel<std::float16_t, std::float16_t, float,   Pow>;
    table[index(float16, float32)] = launchBinaryOpKernel<std::float16_t, float,           float,  Pow>;
    table[index(float32, float16)] = launchBinaryOpKernel<float,           std::float16_t, float,  Pow>;
    table[index(float16, float64)] = launchBinaryOpKernel<std::float16_t, double,          double, Pow>;
    table[index(float64, float16)] = launchBinaryOpKernel<double,          std::float16_t, double, Pow>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<std::float16_t, int8_t,  float,  Pow>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t,         std::float16_t, float,  Pow>;
    table[index(float16, int16)]   = launchBinaryOpKernel<std::float16_t, int16_t, float,  Pow>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t,        std::float16_t, float,  Pow>;
    table[index(float16, int32)]   = launchBinaryOpKernel<std::float16_t, int32_t, float,  Pow>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t,        std::float16_t, float,  Pow>;
    table[index(float16, int64)]   = launchBinaryOpKernel<std::float16_t, int64_t, double, Pow>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t,        std::float16_t, double, Pow>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float,  float,  Pow>;
    table[index(float32, int32)] = launchBinaryOpKernel<float,   int32_t, float,  Pow>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Pow>;
    table[index(float64, int32)] = launchBinaryOpKernel<double,  int32_t, double, Pow>;

    table[index(float32, float32)] = launchBinaryOpKernel<float,  float,  float,  Pow>;
    table[index(float32, float64)] = launchBinaryOpKernel<float,  double, double, Pow>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float,  double, Pow>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Pow>; 

    return table;
}();

} namespace cpu { 

status add(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    return dispatchAdd[index(src1->dtype, src2->dtype)](src1, src2, dst);
}

status sub(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    return dispatchSub[index(src1->dtype, src2->dtype)](src1, src2, dst);
} 

status mul(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    return dispatchMul[index(src1->dtype, src2->dtype)](src1, src2, dst);
}   

status pow(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    return dispatchPow[index(src1->dtype, src2->dtype)](src1, src2, dst);
}

} // namespace cpu