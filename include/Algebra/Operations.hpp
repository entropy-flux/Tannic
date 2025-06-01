#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "Shape.hpp"
#include "Types.hpp"
#include "Expressions.hpp"
#include "Tensor.hpp" 
#include <vector>  

enum class Arity : uint8_t { Unary, Binary };
template<class Operator, Arity arity>  struct Operation;

template<class Operator>
struct Operation<Operator, Arity::Unary> {

    Tensor forward(const Tensor& operand) const{
        Tensor result(operand.shape(), operand.dtype());
        static_cast<const Operator*>(this)->forward(operand, result); 
        return result;
    };
};

struct Negation : public Operation<Negation, Arity::Unary> { 
    using Operation<Negation, Arity::Unary>::forward;
    constexpr type promote(type dtype) const { return dtype; }
    constexpr Shape const& broadcast(Shape const& shape) const { return shape; }
    void forward(Tensor const&, Tensor&) const; 
};

template<class Operator>
struct Operation<Operator, Arity::Binary> { 
    constexpr type promote(type, type) const;
    constexpr Shape broadcast(Shape const&, Shape const&) const;
    
    Tensor forward(Tensor const& operand, Tensor const& cooperand) const {
        if(!(operand.shape() == cooperand.shape())) {
            throw std::runtime_error("Broadcast supported but not implemented for this kernel.");
        }

        type dtype = promote(operand.dtype(), cooperand.dtype());  
        Tensor result(operand.shape(), dtype);
        static_cast<const Operator*>(this)->forward(operand, cooperand, result); 
        return result;
    }
};

struct Addition : public Operation<Addition, Arity::Binary> { 
    using Operation<Addition, Arity::Binary>::forward;
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

struct Subtraction : public Operation<Subtraction, Arity::Binary>  { 
    using Operation<Subtraction, Arity::Binary>::forward;
    void forward(Tensor const&, Tensor const&,  Tensor&) const; 
};

struct Multiplication : public Operation<Multiplication, Arity::Binary>  { 
    using Operation<Multiplication, Arity::Binary>::forward;
    void forward(Tensor const&, Tensor const&, Tensor&) const; 
};

namespace functional{ 
    inline Shape broadcast(Shape const& first, Shape const& second) {
        auto first_rank = first.rank();
        auto second_rank = second.rank();
        auto rank = std::max(first_rank, second_rank);
        std::vector<Shape::size_type> result(rank, 1);

        for (Shape::size_type dimension = 0; dimension < rank; ++dimension) {
            auto first_dimension = (dimension < rank - first_rank) ? 1 : first[dimension - (rank - first_rank)];
            auto second_dimension = (dimension < rank - second_rank) ? 1 : second[dimension - (rank - second_rank)];

            if (first_dimension != second_dimension && first_dimension != 1 && second_dimension != 1) {
                throw "Shapes are not broadcast-compatible.";
            }
            result[dimension] = std::max(first_dimension, second_dimension);
        }
        return Shape(result);
    } 
} 
 
template<class Operator>
constexpr type Operation<Operator, Arity::Binary>::promote(type first, type second) const {
    return static_cast<uint8_t>(first) > static_cast<uint8_t>(second) ? first : second;
}

template<class Operator>
constexpr Shape Operation<Operator, Arity::Binary>::broadcast(Shape const& first, Shape const& second) const {
    return functional::broadcast(first, second);
}
  
template<class Operand>
constexpr auto operator-(Operand const & operand) {
    return Unary<Negation, Operand>{{}, operand};
}

template<class Augend, class Addend>
constexpr auto operator+(Augend const& augend, Addend const& addend) {
    return Binary<Addition, Augend, Addend>{{}, augend, addend};
} 

template<class Subtrahend , class Minuend>
constexpr auto operator-(Subtrahend const& subtrahend, Minuend const& minuend) {
    return Binary<Subtraction, Subtrahend, Minuend>{{}, subtrahend, minuend};
} 

template<class Multiplicand, class Multiplier>
constexpr auto operator*(Multiplicand const& multiplicand, Multiplier const& multiplier) {
    return Binary<Multiplication, Multiplicand, Multiplier>{{}, multiplicand, multiplier};
}  

#endif // OPERATIONS_HPP