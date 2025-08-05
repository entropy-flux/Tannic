#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include "cpu/fns.hpp"  
 
template<typename S, typename D, class Fn>
void scalarFnKernel(
    const S* src_ptr, D* dst_ptr
) {
    Fn fn;
    *dst_ptr = fn(*src_ptr);
}  

template<typename S, typename D, class Fn>
void batchedFnKernel( 
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,           
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides, 
    uint8_t rank, size_t ne
) { 
    Fn fn;  
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

template<typename S, typename D, class Fn>
status launchFnKernel(const tensor_t* src, tensor_t* dst) {
    if (src->rank == 0) {
        scalarFnKernel<S, D, Fn>(
            (const S*)(src->address), 
            (D*)(dst->address)
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
            src->rank, ne
        ); 
    } 
    return SUCCESS;
}     

status launchDefaultKernel(const tensor_t* src, tensor_t* dst) {
    return UNSUPORTED_DTYPE;
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

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}  
 
using Kernel = status(*)(const tensor_t*, tensor_t*);       

constexpr auto dispatchLog = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Log>;
    table[index(float64)] = launchFnKernel<double, double, Log>;
    return table;
}();  

constexpr auto dispatchExp = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Exp>;
    table[index(float64)] = launchFnKernel<double, double, Exp>;
    return table;
}();  
 
constexpr auto dispatchSqrt = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sqrt>;
    table[index(float64)] = launchFnKernel<double, double, Sqrt>;
    return table;
}();

constexpr auto dispatchAbs = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Abs>;
    table[index(float64)] = launchFnKernel<double, double, Abs>;
    table[index(int32)] = launchFnKernel<int32_t, int32_t, Abs>;
    table[index(int64)] = launchFnKernel<int64_t, int64_t, Abs>;
    return table;
}();

constexpr auto dispatchSin = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sin>;
    table[index(float64)] = launchFnKernel<double, double, Sin>;
    return table;
}();

constexpr auto dispatchCos = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Cos>;
    table[index(float64)] = launchFnKernel<double, double, Cos>;
    return table;
}();

constexpr auto dispatchTan = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Tan>;
    table[index(float64)] = launchFnKernel<double, double, Tan>;
    return table;
}();

constexpr auto dispatchSinh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sinh>;
    table[index(float64)] = launchFnKernel<double, double, Sinh>;
    return table;
}();

constexpr auto dispatchCosh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Cosh>;
    table[index(float64)] = launchFnKernel<double, double, Cosh>;
    return table;
}();

constexpr auto dispatchTanh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Tanh>;
    table[index(float64)] = launchFnKernel<double, double, Tanh>;
    return table;
}();

namespace cpu {
 
status log(tensor_t const* src, tensor_t* dst) { 
    return dispatchLog[index(src->dtype)](src, dst);
}

status exp(tensor_t const* src, tensor_t* dst) { 
    return dispatchExp[index(src->dtype)](src, dst);
}

status sqrt(tensor_t const* src, tensor_t* dst) { 
    return dispatchSqrt[index(src->dtype)](src, dst);
}

status abs(tensor_t const* src, tensor_t* dst) { 
    return dispatchAbs[index(src->dtype)](src, dst);
}

status sin(tensor_t const* src, tensor_t* dst) { 
    return dispatchSin[index(src->dtype)](src, dst);
}

status cos(tensor_t const* src, tensor_t* dst) { 
    return dispatchCos[index(src->dtype)](src, dst);
}

status tan(tensor_t const* src, tensor_t* dst) { 
    return dispatchTan[index(src->dtype)](src, dst);
}

status sinh(tensor_t const* src, tensor_t* dst) { 
    return dispatchSinh[index(src->dtype)](src, dst);
}

status cosh(tensor_t const* src, tensor_t* dst) { 
    return dispatchCosh[index(src->dtype)](src, dst);
}

status tanh(tensor_t const* src, tensor_t* dst) { 
    return dispatchTanh[index(src->dtype)](src, dst);
}

} // namespace cpu