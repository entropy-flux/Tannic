#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include <complex> 
#include "cpu/ops.hpp"  
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
    D* dst_ptr, const shape_t& resets, 
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
            shape_t resets{0};
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

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(-float(a));
    }
#endif
}; 

struct Cpy {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(a)) {
        return a;
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return a;
    }
#endif
};

struct Log {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::log(a))) {
        return std::log(a);
    }


#if HAS_FLOAT16        
    half operator()(half a) const noexcept {
        return half(std::log(float(a)));
    }
#endif
};

struct Exp {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::exp(a))) {
        return std::exp(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::exp(float(a)));
    }
#endif
};

struct Sqrt {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sqrt(a))) {
        return std::sqrt(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::sqrt(float(a)));
    }
#endif
};

struct Rsqrt {
    double eps;
    Rsqrt(double e) : eps(e) {}

    template<class A>
    auto operator()(A&& a) const noexcept {
        return 1.0 / std::sqrt(a + eps);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(1.0f / std::sqrt(float(a) + float(eps)));
    }
#endif
};

struct Abs {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::abs(a))) {
        return std::abs(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::fabs(float(a)));
    }
#endif
};

struct Sin {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sin(a))) {
        return std::sin(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::sin(float(a)));
    }
#endif
};

struct Cos {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cos(a))) {
        return std::cos(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::cos(float(a)));
    }
#endif
};

struct Tan {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tan(a))) {
        return std::tan(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::tan(float(a)));
    }
#endif
};

struct Sinh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sinh(a))) {
        return std::sinh(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::sinh(float(a)));
    }
#endif
};

struct Cosh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cosh(a))) {
        return std::cosh(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::cosh(float(a)));
    }
#endif
};

struct Tanh {
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tanh(a))) {
        return std::tanh(a);
    }

#if HAS_FLOAT16    
    half operator()(half a) const noexcept {
        return half(std::tanh(float(a)));
    }
#endif
};

} namespace cpu {

status neg(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:     return launchUnaryOpKernel<int8_t, int8_t, Neg>(src, dst);
        case int16:    return launchUnaryOpKernel<int16_t, int16_t, Neg>(src, dst);
        case int32:    return launchUnaryOpKernel<int32_t, int32_t, Neg>(src, dst);
        case int64:    return launchUnaryOpKernel<int64_t, int64_t, Neg>(src, dst);

#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Neg>(src, dst);
#endif
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
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Cpy>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Cpy>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cpy>(src, dst);
        case complex64:  return launchUnaryOpKernel<std::complex<float>, std::complex<float>, Cpy>(src, dst);
        case complex128: return launchUnaryOpKernel<std::complex<double>, std::complex<double>, Cpy>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status log(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
    
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Log>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Log>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Log>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Exp>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Exp>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Exp>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Sqrt>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Sqrt>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sqrt>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status rsqrt(const tensor_t* src, tensor_t* dst, float eps) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Rsqrt>(src, dst, eps);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Rsqrt>(src, dst, eps);
        case float64:  return launchUnaryOpKernel<double, double, Rsqrt>(src, dst, eps);
        default: return UNSUPPORTED_DTYPE;
    }
}

status abs(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Abs>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Abs>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Abs>(src, dst);
        case int32:    return launchUnaryOpKernel<int32_t, int32_t, Abs>(src, dst);
        case int64:    return launchUnaryOpKernel<int64_t, int64_t, Abs>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Sin>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Sin>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sin>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Cos>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Cos>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cos>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Tan>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Tan>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Tan>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Sinh>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Sinh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Sinh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Cosh>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Cosh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Cosh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
#if HAS_FLOAT16    
        case float16:  return launchUnaryOpKernel<half, half, Tanh>(src, dst);
#endif
        case float32:  return launchUnaryOpKernel<float, float, Tanh>(src, dst);
        case float64:  return launchUnaryOpKernel<double, double, Tanh>(src, dst);
        default: return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu 