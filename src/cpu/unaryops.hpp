#pragma once

#include <array>
#include <cmath>  
#include "core/tensor.h"

namespace cpu {

template<typename TA, typename TB, typename Op>
void unary_op(const tensor_t* A, tensor_t* B, Op op);
 
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

    table[int8] = unary_op<int8_t, int8_t, Negation>;
    table[int16] = unary_op<int16_t, int16_t, Negation>;
    table[int32] = unary_op<int32_t, int32_t, Negation>;
    table[int64] = unary_op<int64_t, int64_t, Negation>;
    table[float32] = unary_op<float, float, Negation>;
    table[float64] = unary_op<double, double, Negation>;

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

    table[float32] = unary_op<float, float, Log>;
    table[float64] = unary_op<double, double, Log>;

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

    table[float32] = unary_op<float, float, Exp>;
    table[float64] = unary_op<double, double, Exp>;

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

    table[float32] = unary_op<float, float, Sqrt>;
    table[float64] = unary_op<double, double, Sqrt>;

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

    table[int8] = unary_op<int8_t, int8_t, Abs>;
    table[int16] = unary_op<int16_t, int16_t, Abs>;
    table[int32] = unary_op<int32_t, int32_t, Abs>;
    table[int64] = unary_op<int64_t, int64_t, Abs>;
    table[float32] = unary_op<float, float, Abs>;
    table[float64] = unary_op<double, double, Abs>;

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

    table[float32] = unary_op<float, float, Sin>;
    table[float64] = unary_op<double, double, Sin>;

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

    table[float32] = unary_op<float, float, Cos>;
    table[float64] = unary_op<double, double, Cos>;

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

    table[float32] = unary_op<float, float, Tan>;
    table[float64] = unary_op<double, double, Tan>;

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

    table[float32] = unary_op<float, float, Sinh>;
    table[float64] = unary_op<double, double, Sinh>;

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

    table[float32] = unary_op<float, float, Cosh>;
    table[float64] = unary_op<double, double, Cosh>;

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

    table[float32] = unary_op<float, float, Tanh>;
    table[float64] = unary_op<double, double, Tanh>;

    return table;
}();

} // namespace cpu::tanh