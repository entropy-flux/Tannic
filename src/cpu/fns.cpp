#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include "cpu/fns.hpp"  
 
namespace { 

struct Idn { 
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

template<typename S, typename D, class Fn>
void scalarFnKernel(
    const S* src_ptr, D* dst_ptr, Fn fn
) {
    *dst_ptr = fn(*src_ptr);
}    

template<typename S, typename D, class Fn>
void batchedFnKernel( 
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,           
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides, 
    uint8_t rank, size_t ne, Fn fn
) {  
    size_t cnt[8] = {0};
    for (size_t idx = 0; idx < ne; ++idx) {
        size_t offs = 0;
        for (int dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_strides.sizes[dim];
        }

        dst_ptr[idx] = fn(src_ptr[offs]);

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_shape.sizes[dim])
                break;
            cnt[dim] = 0;
        }
    } 
}

template<typename S, typename D, class Fn, class ... Args>
status launchFnKernel(const tensor_t* src, tensor_t* dst, Args... args) {
     Fn fn(std::forward<Args>(args)...);

    if (src->rank == 0) {
        scalarFnKernel<S, D, Fn>(
            (const S*)(src->address), 
            (D*)(dst->address), fn
        ); 
    } 
    
    else {    
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        batchedFnKernel<S, D, Fn>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne, fn
        ); 
    } 
    return SUCCESS;
}        

} namespace cpu {

status idn(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case int8:
            return launchFnKernel<int8_t, int8_t, Idn>(src, dst);
        case int16:
            return launchFnKernel<int16_t, int16_t, Idn>(src, dst); 
        case int32:
            return launchFnKernel<int32_t, int32_t, Idn>(src, dst); 
        case int64:
            return launchFnKernel<int64_t, int64_t, Idn>(src, dst); 
        case float32:
            return launchFnKernel<float, float, Idn>(src, dst);
        case float64:
            return launchFnKernel<double, double, Idn>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
} 

status log(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Log>(src, dst);
        case float64:
            return launchFnKernel<double, double, Log>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Exp>(src, dst);
        case float64:
            return launchFnKernel<double, double, Exp>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sqrt>(src, dst);
        case float64:
            return launchFnKernel<double, double, Sqrt>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status rsqrt(const tensor_t* src, tensor_t* dst, float eps) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Rsqrt>(src, dst, eps);
        case float64:
            return launchFnKernel<double, double, Rsqrt>(src, dst, eps);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status abs(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Abs>(src, dst);
        case float64:
            return launchFnKernel<double, double, Abs>(src, dst);
        case int32:
            return launchFnKernel<int32_t, int32_t, Abs>(src, dst);
        case int64:
            return launchFnKernel<int64_t, int64_t, Abs>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sin>(src, dst);
        case float64:
            return launchFnKernel<double, double, Sin>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Cos>(src, dst);
        case float64:
            return launchFnKernel<double, double, Cos>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Tan>(src, dst);
        case float64:
            return launchFnKernel<double, double, Tan>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sinh>(src, dst);
        case float64:
            return launchFnKernel<double, double, Sinh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Cosh>(src, dst);
        case float64:
            return launchFnKernel<double, double, Cosh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Tanh>(src, dst);
        case float64:
            return launchFnKernel<double, double, Tanh>(src, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu