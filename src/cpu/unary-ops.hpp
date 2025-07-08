#pragma once

#include <array>
#include <cmath>  
#include "cpu/cpu.hpp"
#include "core/tensor.h"


namespace {

[[noreturn]] inline void notImplemented(const tensor_t*, tensor_t*, auto) {
    throw std::runtime_error("Kernel not implemented for this type");
} 

}

namespace cpu {

template<typename S, typename D, typename Op>
void unaryOp(const tensor_t* A, tensor_t* B, Op op);
 
} // namespace cpu

namespace cpu::negation {

class Negation {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(-a)) -> decltype(-a) {
        return -a;
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Negation);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented); 
    table[int8] = unaryOp<int8_t, int8_t, Negation>;
    table[int16] = unaryOp<int16_t, int16_t, Negation>;
    table[int32] = unaryOp<int32_t, int32_t, Negation>;
    table[int64] = unaryOp<int64_t, int64_t, Negation>;
    table[float32] = unaryOp<float, float, Negation>;
    table[float64] = unaryOp<double, double, Negation>;

    return table;
}();

} // namespace cpu::negation


namespace cpu::log {

class Log {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::log(a))) {
        return std::log(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Log);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{}; 
    table.fill(::notImplemented);
    table[float32] = unaryOp<float, float, Log>;
    table[float64] = unaryOp<double, double, Log>;

    return table;
}();

} // namespace cpu::log 



namespace cpu::exp {

class Exp {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::exp(a))) {
        return std::exp(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Exp);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented); 
    table[float32] = unaryOp<float, float, Exp>;
    table[float64] = unaryOp<double, double, Exp>;

    return table;
}();

} // namespace cpu::exp



namespace cpu::sqrt {

class Sqrt {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::sqrt(a))) {
        return std::sqrt(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sqrt);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented); 
    table[float32] = unaryOp<float, float, Sqrt>;
    table[float64] = unaryOp<double, double, Sqrt>;

    return table;
}();

} // namespace cpu::sqrt


namespace cpu::abs {

class Abs {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::abs(a))) {
        return std::abs(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Abs);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented); 
    table[int8] = unaryOp<int8_t, int8_t, Abs>;
    table[int16] = unaryOp<int16_t, int16_t, Abs>;
    table[int32] = unaryOp<int32_t, int32_t, Abs>;
    table[int64] = unaryOp<int64_t, int64_t, Abs>;
    table[float32] = unaryOp<float, float, Abs>;
    table[float64] = unaryOp<double, double, Abs>;

    return table;
}();

} // namespace cpu::abs


namespace cpu::sin {

class Sin {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::sin(a))) {
        return std::sin(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sin);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented);  
    table[float32] = unaryOp<float, float, Sin>;
    table[float64] = unaryOp<double, double, Sin>;

    return table;
}();

} // namespace cpu::sin



namespace cpu::cos {

class Cos {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::cos(a))) {
        return std::cos(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Cos);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented); 
    table[float32] = unaryOp<float, float, Cos>;
    table[float64] = unaryOp<double, double, Cos>;

    return table;
}();

} // namespace cpu::cos



namespace cpu::tan {

class Tan {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::tan(a))) {
        return std::tan(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Tan);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{}; 
    table.fill(::notImplemented);
    table[float32] = unaryOp<float, float, Tan>;
    table[float64] = unaryOp<double, double, Tan>;

    return table;
}();

} // namespace cpu::tan



namespace cpu::sinh {

class Sinh {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::sinh(a))) {
        return std::sinh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Sinh);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{}; 
    table.fill(::notImplemented);
    table[float32] = unaryOp<float, float, Sinh>;
    table[float64] = unaryOp<double, double, Sinh>;

    return table;
}();

} // namespace cpu::sinh

 

namespace cpu::cosh {

class Cosh {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::cosh(a))) {
        return std::cosh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Cosh);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{};
    table.fill(::notImplemented);
    table[float32] = unaryOp<float, float, Cosh>;
    table[float64] = unaryOp<double, double, Cosh>;

    return table;
}();

} // namespace cpu::cosh



namespace cpu::tanh {

class Tanh {
public:
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(std::tanh(a))) {
        return std::tanh(a);
    }
};

using Kernel = void(*)(const tensor_t*, tensor_t*, Tanh);

constexpr auto kernels = []() {
    std::array<Kernel, TYPES> table{}; 
    table.fill(::notImplemented);
    table[float32] = unaryOp<float, float, Tanh>;
    table[float64] = unaryOp<double, double, Tanh>;

    return table;
}();

} // namespace cpu::tanh