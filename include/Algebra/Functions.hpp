#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include "Tensor.hpp"
#include "Expressions.hpp"

template<class Function>
struct Functor {
    constexpr type promote(type dtype) const { return dtype; }
    constexpr Shape const& broadcast(Shape const& shape) const { return shape; }
    Tensor forward(const Tensor& argument) const {
        Tensor result(argument.shape(), argument.dtype());
        static_cast<const Function*>(this)->forward(argument, result);
        return result;
    }
};

struct Log : public Functor<Log> {
    uint8_t base = 0;
    void forward(Tensor const&, Tensor&) const;
};

struct Exp : public Functor<Exp> {
    uint8_t base = 0;
    void forward(Tensor const&, Tensor&) const;
};

struct Sqrt : public Functor<Sqrt> {
    void forward(Tensor const&, Tensor&) const;
};

struct Abs : public Functor<Abs> {
    void forward(Tensor const&, Tensor&) const;
};

struct Sin : public Functor<Sin> {
    void forward(Tensor const&, Tensor&) const;
};

struct Sinh : public Functor<Sinh> {
    void forward(Tensor const&, Tensor&) const;
};

struct Cos : public Functor<Cos> {
    void forward(Tensor const&, Tensor&) const;
};

struct Cosh : public Functor<Cosh> {
    void forward(Tensor const&, Tensor&) const;
};

struct Tan : public Functor<Tan> {
    void forward(Tensor const&, Tensor&) const;
};

struct Tanh : public Functor<Tanh> {
    void forward(Tensor const&, Tensor&) const;
}; 

template<class Operand>
constexpr auto log(Operand const& operand) {
    return Unary<Log, Operand>{{}, operand};
}

template<class Operand>
constexpr auto exp(Operand const& operand) {
    return Unary<Exp, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sqrt(Operand const& operand) {
    return Unary<Sqrt, Operand>{{}, operand};
}

template<class Operand>
constexpr auto abs(Operand const& operand) {
    return Unary<Abs, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sin(Operand const& operand) {
    return Unary<Sin, Operand>{{}, operand};
}

template<class Operand>
constexpr auto sinh(Operand const& operand) {
    return Unary<Sinh, Operand>{{}, operand};
}

template<class Operand>
constexpr auto cos(Operand const& operand) {
    return Unary<Cos, Operand>{{}, operand};
}

template<class Operand>
constexpr auto cosh(Operand const& operand) {
    return Unary<Cosh, Operand>{{}, operand};
}

template<class Operand>
constexpr auto tan(Operand const& operand) {
    return Unary<Tan, Operand>{{}, operand};
}

template<class Operand>
constexpr auto tanh(Operand const& operand) {
    return Unary<Tanh, Operand>{{}, operand};
}

#endif // FUNCTIONS_HPP