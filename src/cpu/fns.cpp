#include "cpu.hpp"  
#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  

template<typename S, typename D, class F>
void fnKernel(
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    F func{};
    const S* src = static_cast<const S*>(src_ptr);
    D* dst = static_cast<D*>(dst_ptr);

    if (rank == 0) {
        *dst = func(*src);
        return;
    }

    size_t total = 1;
    for (uint8_t i = 0; i < rank; ++i)
        total *= dst_sz[i];

    for (size_t idx = 0; idx < total; ++idx) {
        size_t offs = 0;
        for (uint8_t dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_ne[dim];
        }

        dst[idx] = func(src[offs]);
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_sz[dim])
                break;
            cnt[dim] = 0;
        }
    }
}     
 
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

namespace fn {
 
using Kernel = void(*)(
    const void* src, const size_t* src_sz, const size_t* src_ne,
    void* dst, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
);      

constexpr void defaultKernel(
    const void* src, const size_t* src_sz, const size_t* src_ne,
    void* dst, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    throw std::runtime_error("Not supported dtype");
}; 

constexpr auto log = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Log>;
    table[index(float64)] = fnKernel<double, double, Log>;
    return table;
}();  

constexpr auto exp = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Exp>;
    table[index(float64)] = fnKernel<double, double, Exp>;
    return table;
}();  
 
constexpr auto sqrt = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Sqrt>;
    table[index(float64)] = fnKernel<double, double, Sqrt>;
    return table;
}();

constexpr auto abs = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Abs>;
    table[index(float64)] = fnKernel<double, double, Abs>;
    table[index(int32)] = fnKernel<int32_t, int32_t, Abs>;
    table[index(int64)] = fnKernel<int64_t, int64_t, Abs>;
    return table;
}();

constexpr auto sin = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Sin>;
    table[index(float64)] = fnKernel<double, double, Sin>;
    return table;
}();

constexpr auto cos = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Cos>;
    table[index(float64)] = fnKernel<double, double, Cos>;
    return table;
}();

constexpr auto tan = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Tan>;
    table[index(float64)] = fnKernel<double, double, Tan>;
    return table;
}();

constexpr auto sinh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Sinh>;
    table[index(float64)] = fnKernel<double, double, Sinh>;
    return table;
}();

constexpr auto cosh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Cosh>;
    table[index(float64)] = fnKernel<double, double, Cosh>;
    return table;
}();

constexpr auto tanh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(defaultKernel);
    table[index(float32)] = fnKernel<float, float, Tanh>;
    table[index(float64)] = fnKernel<double, double, Tanh>;
    return table;
}();

} namespace cpu {

void log(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::log[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void exp(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::exp[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void sqrt(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::sqrt[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void abs(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::abs[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void sin(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::sin[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void cos(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::cos[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void tan(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::tan[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void sinh(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::sinh[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void cosh(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::cosh[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void tanh(tensor_t const* src, tensor_t* dst) {
    std::vector<size_t> cnt(src->rank, 0);
    fn::tanh[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

} // namespace cpu