#ifndef EXPRESSIONS_HPP
#define EXPRESSIONS_HPP

#include "Types.hpp"
#include "Shape.hpp" 
#include "Tensor.hpp"
#include "Parameter.hpp"

namespace expression {

    template<typename T>
    struct Trait {
        using Reference = T;
    };

    template<>
    struct Trait<Tensor> {
        using Reference = Tensor const &;
    };

    template<class Operation, class Operand>
    struct Unary { 
        Operation operation;
        Trait<Operand>::Reference operand;  

        constexpr Unary(Operation operation, Trait<Operand>::Reference operand)
        :   operation(operation)
        ,   operand(operand) 
        {}

        constexpr decltype(auto) shape() const {
            return operation.broadcast(operand.shape());
        }

        constexpr decltype(auto) dtype() const {
            return operation.promote(operand.dtype());
        }

        Tensor forward() const {
            Tensor result(shape(), dtype());
            operation.perform(operand.forward().view(), result);
            return result;
        }
    };


    template<class Operation, class Operand, class Cooperand>
    struct Binary {  
        Operation operation;
        Trait<Operand>::Reference operand;  
        Trait<Cooperand>::Reference cooperand;

        constexpr Binary(Operation operation, Trait<Operand>::Reference operand, Trait<Cooperand>::Reference cooperand)
        :   operation(operation)
        ,   operand(operand)
        ,   cooperand(cooperand) 
        {}  

        constexpr decltype(auto) shape() const {
            return operation.broadcast(operand.shape(), cooperand.shape());
        }

        constexpr decltype(auto) dtype() const {
            return operation.promote(operand.dtype(), cooperand.dtype());
        }

        Tensor forward() const {
            Tensor result(shape(), dtype());
            operation.perform(operand.forward().view(), cooperand.forward().view(), result);
            return result;
        }
    };

} // namespace expression 

#endif // EXPRESSIONS_HPP