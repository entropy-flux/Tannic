#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <vector> 

#include "Shape.hpp"
#include "Types.hpp"
#include "Expressions.hpp"

class Tensor;
class View;
 
struct Operation {

    constexpr type promote(type only) const { 
        return only; 
    } 

    constexpr type promote(type first, type second) const {
        return first;
    }

    constexpr const Shape& broadcast(Shape const& only) const {
        return only;
    }

    constexpr static Shape broadcast(Shape const& first, Shape const& second) {
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

};


namespace operation {
    
    struct Negation : public Operation { 
        void perform(View const& operand, Tensor& result) const;
    };


    struct Addition : public Operation { 
        void perform(View const& operand, View const& cooperand, Tensor& result) const;
    };


    struct Subtraction : public Operation {  
        void perform(View const& operand, View const& cooperand, Tensor& result) const;
    };


    struct Multiplication : public Operation {  
        void perform(View const& operand, View const& cooperand, Tensor& result) const;
    };


    struct Matmul {
        
        constexpr static Shape broadcast(Shape const& first, Shape const& second) {
            size_t first_rank = first.rank();
            size_t second_rank = second.rank();
            
            if (first_rank == 1 && second_rank == 1) {
                if (first[0] != second[0])
                    throw "Vector sizes must match for dot product";
                return Shape{first[0]}; // scalar
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
            Shape batches = Operation::broadcast(first_batches, second_batches);

            auto a_m = *(first.end() - 2);
            auto a_k = *(first.end() - 1);
            auto b_k = *(second.end() - 2);
            auto b_n = *(second.end() - 1);

            if (*(first.end() - 1) != *(second.end() - 2))
                throw std::invalid_argument("Inner dimensions must match for matmul.");

            std::vector<Shape::size_type> result(batches.begin(), batches.end());
            result.push_back(*(first.end() - 2));
            result.push_back(*(second.end() - 1));
            return Shape(result); 
        };
    };

} 

template<class Operand>
constexpr auto operator-(Operand const & operand) {
    return expression::Unary<operation::Negation, Operand>{{}, operand};
}

template<class Augend, class Addend>
constexpr auto operator+(Augend const& augend, Addend const& addend) {
    return expression::Binary<operation::Addition, Augend, Addend>{{}, augend, addend};
} 

template<class Subtrahend , class Minuend>
constexpr auto operator-(Subtrahend const& subtrahend, Minuend const& minuend) {
    return expression::Binary<operation::Subtraction, Subtrahend, Minuend>{{}, subtrahend, minuend};
} 

template<class Multiplicand, class Multiplier>
constexpr auto operator*(Multiplicand const& multiplicand, Multiplier const& multiplier) {
    return expression::Binary<operation::Multiplication, Multiplicand, Multiplier>{{}, multiplicand, multiplier};
}

#endif // OPERATIONS_HPP