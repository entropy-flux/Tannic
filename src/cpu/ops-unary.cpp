#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include <complex>
#include <stdfloat>
#include "cpu/ops.hpp"  
 
namespace {  

template<typename S, typename D, class Op>
void singletonUnaryOpKernel(
    const S* src_ptr, D* dst_ptr, Op op
) {
    *dst_ptr = op(*src_ptr);
}    

template<typename S, typename D, class Op>
void contiguousUnaryOpKernel(
    const S* src_ptr, D* dst_ptr, size_t ne, Op op
) {
    for (size_t i = 0; i < ne; ++i) {
        dst_ptr[i] = op(src_ptr[i]);
    }
}

template<typename S, typename D, class Op>
void stridedUnaryOpKernel( 
    const S* src_ptr, const strides_t& src_strides,           
    D* dst_ptr, const strides_t& resets, 
    uint8_t rank, size_t ne, Op op
) {  
    size_t current_offset = 0;  
    for (size_t idx = 0; idx < ne; ++idx) {
        dst_ptr[idx] = op(src_ptr[current_offset]); 
        current_offset += src_strides.sizes[rank - 1];
         
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (current_offset % resets.sizes[dim] != 0)
                break;
             
            current_offset -= resets.sizes[dim];
            if (dim > 0) {
                current_offset += src_strides.sizes[dim - 1];
            }
        }
    } 
}

template<typename S, typename D, class Op, class ... Args>
status launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, Args... args) {
    Op op(std::forward<Args>(args)...);
    size_t ne = dst->size;

    switch (src->layout) {
        case SINGLETON: {
            singletonUnaryOpKernel<S, D, Op>(
                (const S*)(src->address),
                (D*)(dst->address),
                op
            );
            return SUCCESS;
        }

        case CONTIGUOUS: {
                contiguousUnaryOpKernel<S, D, Op>(
                    (const S*)(src->address),
                    (D*)(dst->address),
                    ne,
                    op
                ); 
            return SUCCESS;
        }

        case STRIDED: {   
            strides_t strides{0};
            strides_t resets{0};
            for (int dim = 0; dim < src->rank; ++dim) {
                resets.sizes[dim] = dst->shape.sizes[dim] * src->strides.sizes[dim];
                strides.sizes[dim] = src->strides.sizes[dim];
            }

            stridedUnaryOpKernel<S, D, Op>(
                (const S*)(src->address), strides,
                (D*)(dst->address), resets,
                src->rank, ne,
                op
            );
            return SUCCESS;
        }

        default:
            return ERROR; 
    } 
} 

struct Neg {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(-a)) {
        return -a;
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(-float(a));
    }
};

struct Cpy {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(a)) {
        return a;
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return a;
    }
};

struct Log {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::log(a))) {
        return std::log(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::log(float(a)));
    }
};

struct Exp {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::exp(a))) {
        return std::exp(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::exp(float(a)));
    }
};

struct Sqrt {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sqrt(a))) {
        return std::sqrt(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::sqrt(float(a)));
    }
};

struct Rsqrt {
    double eps;
    Rsqrt(double e) : eps(e) {}

    template<class A>
    auto operator()(A&& a) const noexcept {
        return 1.0 / std::sqrt(a + eps);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(1.0f / std::sqrt(float(a) + float(eps)));
    }
};

struct Abs {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::abs(a))) {
        return std::abs(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::fabs(float(a)));
    }
};

struct Sin {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sin(a))) {
        return std::sin(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::sin(float(a)));
    }
};

struct Cos {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cos(a))) {
        return std::cos(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::cos(float(a)));
    }
};

struct Tan {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tan(a))) {
        return std::tan(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::tan(float(a)));
    }
};

struct Sinh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sinh(a))) {
        return std::sinh(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::sinh(float(a)));
    }
};

struct Cosh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cosh(a))) {
        return std::cosh(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::cosh(float(a)));
    }
};

struct Tanh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tanh(a))) {
        return std::tanh(a);
    }
    std::float16_t operator()(std::float16_t a) const noexcept {
        return std::float16_t(std::tanh(float(a)));
    }
};

} namespace cpu {

status neg(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:     return launchUnaryOpKernel<int8_t, int8_t, Neg>(src, dst);
        case int16:    return launchUnaryOpKernel<int16_t, int16_t, Neg>(src, dst);
        case int32:    return launchUnaryOpKernel<int32_t, int32_t, Neg>(src, dst);
        case int64:    return launchUnaryOpKernel<int64_t, int64_t, Neg>(src, dst);
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Neg>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Neg>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Neg>(src, dst);
        case complex64:  return launchUnaryOpKernel<std::complex<float>, std::complex<float>, Neg>(src, dst);
        case complex128: return launchUnaryOpKernel<std::complex<double>, std::complex<double>, Neg>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status cpy(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:     return launchUnaryOpKernel<int8_t, int8_t, Cpy>(src, dst);
        case int16:    return launchUnaryOpKernel<int16_t, int16_t, Cpy>(src, dst);
        case int32:    return launchUnaryOpKernel<int32_t, int32_t, Cpy>(src, dst);
        case int64:    return launchUnaryOpKernel<int64_t, int64_t, Cpy>(src, dst);
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Cpy>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Cpy>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cpy>(src, dst);
        case complex64:  return launchUnaryOpKernel<std::complex<float>, std::complex<float>, Cpy>(src, dst);
        case complex128: return launchUnaryOpKernel<std::complex<double>, std::complex<double>, Cpy>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status log(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Log>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Log>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Log>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Exp>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Exp>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Exp>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Sqrt>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Sqrt>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sqrt>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status rsqrt(const tensor_t* src, tensor_t* dst, float eps) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Rsqrt>(src, dst, eps);
        case float32:  return launchUnaryOpKernel<float, float, Rsqrt>(src, dst, eps);
        case float64:  return launchUnaryOpKernel<double, double, Rsqrt>(src, dst, eps);
        default: return UNSUPPORTED_DTYPE;
    }
}

status abs(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Abs>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Abs>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Abs>(src, dst);
        case int32:    return launchUnaryOpKernel<int32_t, int32_t, Abs>(src, dst);
        case int64:    return launchUnaryOpKernel<int64_t, int64_t, Abs>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Sin>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Sin>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sin>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Cos>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Cos>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cos>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Tan>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Tan>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Tan>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Sinh>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Sinh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sinh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Cosh>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Cosh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cosh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float16:  return launchUnaryOpKernel<std::float16_t, std::float16_t, Tanh>(src, dst);
        case float32:  return launchUnaryOpKernel<float, float, Tanh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Tanh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu 