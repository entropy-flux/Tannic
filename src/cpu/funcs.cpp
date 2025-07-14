#include <vector>
#include "cpu/funcs.hpp"   

template<typename S, typename D, class F>
void func(
    const S* src, const size_t* src_sz, const size_t* src_ne,
    D* dst, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt, F func
) {
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
   
namespace cpu {  
 
template<Function F>
struct Functor;

template<typename S, typename T, Function F>
bool function(const tensor_t* src, tensor_t* dst) { 
    uint8_t rank = src->rank;
    std::vector<std::size_t> indexes(rank, 0);
    func<S, T, Functor<F>>(
        static_cast<const S*>(src->address), src->shape, src->strides,
        static_cast<T*>(dst->address), dst->shape, dst->strides,
        rank, indexes.data(), {}
    ); 
    return true;
} 

template<>
struct Functor<Function::LOG> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::log(a))) {
        return std::log(a);
    }
};

template<>
struct Functor<Function::EXP> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::exp(a))) {
        return std::exp(a);
    }
};

template<>
struct Functor<Function::TAN> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tan(a))) {
        return std::tan(a);
    }
};

template<>
struct Functor<Function::SQRT> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sqrt(a))) {
        return std::sqrt(a);
    }
};

template<>
struct Functor<Function::ABS> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::abs(a))) {
        return std::abs(a);
    }
};

template<>
struct Functor<Function::SIN> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sin(a))) {
        return std::sin(a);
    }
};

template<>
struct Functor<Function::COS> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cos(a))) {
        return std::cos(a);
    }
};

template<>
struct Functor<Function::SINH> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::sinh(a))) {
        return std::sinh(a);
    }
};

template<>
struct Functor<Function::COSH> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::cosh(a))) {
        return std::cosh(a);
    }
};

template<>
struct Functor<Function::TANH> { 
    template<class A>
    auto operator()(A&& a) const noexcept(noexcept(std::tanh(a))) {
        return std::tanh(a);
    }
};  
  
template bool function<float, float, Function::LOG>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::LOG>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::EXP>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::EXP>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::SQRT>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::SQRT>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::ABS>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::ABS>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::SIN>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::SIN>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::COS>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::COS>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::TAN>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::TAN>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::SINH>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::SINH>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::COSH>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::COSH>(const tensor_t*, tensor_t*);

template bool function<float, float, Function::TANH>(const tensor_t*, tensor_t*);
template bool function<double, double, Function::TANH>(const tensor_t*, tensor_t*); 

}  