#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "Algebra/Operations.hpp"

struct Matmul {
    constexpr type promote(type, type) const;
    constexpr Shape broadcast(Shape const&, Shape const&) const;
    void forward(Tensor const&, Tensor const&, Tensor&) const;

    Tensor forward(Tensor const& operand, Tensor const& cooperand) const { 
        type dtype = promote(operand.dtype(), cooperand.dtype());   
        Tensor result(Matmul::broadcast(operand.shape(), cooperand.shape()), dtype);
        forward(operand, cooperand, result);
        return result;
    }
}; 

constexpr type Matmul::promote(type first, type second) const {
    return static_cast<uint8_t>(first) > static_cast<uint8_t>(second) ? first : second;
}

constexpr Shape Matmul::broadcast(Shape const& first, Shape const& second) const {
    auto first_rank = first.rank();
    auto second_rank = second.rank();

    if (first_rank == 1 && second_rank == 1) {
        if (first[0] != second[0])
            throw "Vector sizes must match for dot product";
        return Shape{first[0]};
    }

    if (first_rank == 2 && second_rank == 1) {
        if (first[1] != second[0])
            throw std::invalid_argument("Matrix inner dimensions do not match.");
        return Shape{first[0]};
    }

    if (first_rank == 1 && second_rank == 2) {
        if (first[0] != second[0])
            throw std::invalid_argument("Matrix inner dimensions do not match.");
        return Shape{second[1]};
    }

    Shape first_batches(first.begin(), first.end() - 2);
    Shape second_batches(second.begin(), second.end() - 2);
    Shape batches = functional::broadcast(first_batches, second_batches);

    if (*(first.end() - 1) != *(second.end() - 2))
        throw std::invalid_argument("Inner dimensions must match for matmul.");

    std::vector<Shape::size_type> result(batches.begin(), batches.end());
    result.push_back(*(first.end() - 2));
    result.push_back(*(second.end() - 1));
    return Shape(result);
}

template<class Multiplicand, class Multiplier>
constexpr auto matmul(Multiplicand const& multiplicand, Multiplier const& multiplier) {
    return Binary<Matmul, Multiplicand, Multiplier>{{}, multiplicand, multiplier};
}


/*

Works but unimplemented.
struct Linear {
    constexpr static decltype(auto) promote(type, type, type); 
    constexpr static decltype(auto) broadcast(Shape const&, Shape const&, Shape const&);
    static Tensor forward(Tensor const&, Tensor const&, Tensor const&);
} ;

template<class Multiplicand, class Multiplier, class Addend>
auto operator+(Binary<Matmul, Multiplicand, Multiplier> const& augend, Addend const& addend) {
    return Variadic<Linear, Multiplicand, Multiplier, Addend>(Linear{}, augend.operand, augend.cooperand, addend);
}

*/

#endif // TRANSFORMATIONS_HPP