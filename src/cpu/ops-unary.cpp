#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include <complex>
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
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,           
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides, 
    uint8_t rank, size_t ne, Op op
) {  
    size_t cnt[8] = {0}; 
    for (size_t idx = 0; idx < ne; ++idx) {
        size_t offs = 0;
        for (int dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_strides.sizes[dim];
        }

        dst_ptr[idx] = op(src_ptr[offs]);

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_shape.sizes[dim])
                break;
            cnt[dim] = 0;
        }
    } 
} 

template<typename S, typename D, class Op, class ... Args>
status launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, Args... args) {
    Op op(std::forward<Args>(args)...);
    size_t ne = dst->size;

    switch (src->layout) {
        case SINGLETON:
            singletonUnaryOpKernel<S, D, Op>(
                (const S*)(src->address),
                (D*)(dst->address),
                op
            );
            break;

        case CONTIGUOUS:
            contiguousUnaryOpKernel<S, D, Op>(
                (const S*)(src->address),
                (D*)(dst->address),
                ne,
                op
            );
            break;

        case STRIDED:
            stridedUnaryOpKernel<S, D, Op>(
                (const S*)(src->address), src->shape, src->strides,
                (D*)(dst->address), dst->shape, dst->strides,
                src->rank, ne,
                op
            );
            break;

        default:
            return ERROR; 
    }

    return SUCCESS;
}


struct Neg { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(-a)) {
        return -a;
    }
};

struct Cpy { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(a)) {
        return a;
    }
};

struct Log { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::log(a))) {
        return std::log(a);
    }
};
 
struct Exp { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::exp(a))) {
        return std::exp(a);
    }
};
  
struct Sqrt { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sqrt(a))) {
        return std::sqrt(a);
    }
}; 

struct Rsqrt {
    double eps;
    Rsqrt(double e) : eps(e) {}

    template<class A>
    auto operator()(A&& a) const noexcept {
        return 1.0 / std::sqrt(a + eps);
    }
}; 
 
struct Abs { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::abs(a))) {
        return std::abs(a);
    }
};
 
struct Sin { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sin(a))) {
        return std::sin(a);
    }
};
 
struct Cos { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cos(a))) {
        return std::cos(a);
    }
};

struct Tan { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tan(a))) {
        return std::tan(a);
    }
}; 

struct Sinh { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sinh(a))) {
        return std::sinh(a);
    }
};
 
struct Cosh{ 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cosh(a))) {
        return std::cosh(a);
    }
};
 
struct Tanh { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tanh(a))) {
        return std::tanh(a);
    }
};    

} namespace cpu { 

status neg(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:
            return launchUnaryOpKernel<int8_t, int8_t, Neg>(src, dst);
        case int16:
            return launchUnaryOpKernel<int16_t, int16_t, Neg>(src, dst);
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Neg>(src, dst);
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Neg>(src, dst);
        case float32:
            return launchUnaryOpKernel<float, float, Neg>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Neg>(src, dst);
        case complex64:
            return launchUnaryOpKernel<std::complex<float>, std::complex<float>, Neg>(src, dst);
        case complex128:
            return launchUnaryOpKernel<std::complex<double>, std::complex<double>, Neg>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cpy(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:
            return launchUnaryOpKernel<int8_t, int8_t, Cpy>(src, dst);
        case int16:
            return launchUnaryOpKernel<int16_t, int16_t, Cpy>(src, dst); 
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Cpy>(src, dst); 
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Cpy>(src, dst); 
        case float32:
            return launchUnaryOpKernel<float, float, Cpy>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Cpy>(src, dst);
        case complex64:
            return launchUnaryOpKernel<std::complex<float>, std::complex<float>, Cpy>(src, dst);
        case complex128:
            return launchUnaryOpKernel<std::complex<double>, std::complex<double>, Cpy>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
} 

status log(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Log>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Log>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Exp>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Exp>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Sqrt>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Sqrt>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status rsqrt(const tensor_t* src, tensor_t* dst, float eps) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Rsqrt>(src, dst, eps);
        case float64:
            return launchUnaryOpKernel<double, double, Rsqrt>(src, dst, eps);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status abs(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Abs>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Abs>(src, dst);
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Abs>(src, dst);
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Abs>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Sin>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Sin>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Cos>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Cos>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Tan>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Tan>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Sinh>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Sinh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Cosh>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Cosh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchUnaryOpKernel<float, float, Tanh>(src, dst);
        case float64:
            return launchUnaryOpKernel<double, double, Tanh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
} 

} // namespace cpu