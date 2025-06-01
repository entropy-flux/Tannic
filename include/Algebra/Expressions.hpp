#ifndef EXPRESSIONS_HPP
#define EXPRESSIONS_HPP

#include "Types.hpp"
#include "Shape.hpp"   

template<typename T>
struct Trait {
    using Reference = T;
};
 
struct Fusable {
    constexpr static bool is_fusable = true;
};  

template<class Operation, class Operand>
struct Unary { 
    Operation operation;
    typename Trait<Operand>::Reference operand; 

    constexpr Unary(Operation operation, typename Trait<Operand>::Reference operand)
    :   operation(operation)
    ,   operand(operand) 
    {}

    constexpr auto shape() const -> decltype(auto) {
        return operation.broadcast(operand.shape());
    }

    constexpr auto dtype() const -> decltype(auto)  {
        return operation.promote(operand.dtype());
    }

    auto forward() const {   
        return operation.forward(operand.forward()); 
    }
};
 

template<class Operation, class Operand, class Cooperand>
struct Binary {  
    Operation operation;
    typename Trait<Operand>::Reference operand; 
    typename Trait<Cooperand>::Reference cooperand; 

    constexpr Binary(Operation operation, typename Trait<Operand>::Reference operand, typename Trait<Cooperand>::Reference cooperand)
    :   operation(operation)
    ,   operand(operand)
    ,   cooperand(cooperand) 
    {}  

    constexpr auto shape() const -> decltype(auto)  {
        return operation.broadcast(operand.shape(), cooperand.shape());
    }

    constexpr auto dtype() const -> decltype(auto) {
        return operation.promote(operand.dtype(), cooperand.dtype());
    }

    auto forward() const -> decltype(auto)  {  
        return operation.forward(operand.forward(), cooperand.forward()); 
    }
};


template<class Operation, class... Operands>
struct Variadic {
    Operation operation;
    std::tuple<typename Trait<Operands>::Reference...> operands;

    constexpr Variadic(Operation operation, typename Trait<Operands>::Reference... operands)
        : operation(operation), operands(std::forward<decltype(operands)>(operands)...) {}

    constexpr auto shape() const  -> decltype(auto) {
        return std::apply([&](const auto&... arguments) {
            return operation.broadcast(arguments.shape()...);
        }, operands);
    }

    constexpr auto dtype() const  -> decltype(auto) {
        return std::apply([&](const auto&... arguments) {
            return operation.promote(arguments.dtype()...);
        }, operands);
    }

    auto forward() const -> decltype(auto) {
        return std::apply([&](const auto&... arguments) {
            return operation.forward(arguments.forward()...);
        }, operands);
    }
}; 

/* Untested optimization. 
template<class Operation, class Prior, class Operand> requires (Operation::is_fusable)
struct Unary<Operation, Unary<Prior, Operand>> {
    Operation operation;
    Unary<Prior, Operand> prior; 

    constexpr Unary(Operation operation, Unary<Prior, Operand> prior)
    :   operation(operation)
    ,   prior(prior) 
    {}
    
    constexpr auto shape() const -> decltype(auto) {
        return prior.operation.broadcast(prior.operand.shape());
    }

    constexpr auto dtype() const -> decltype(auto)  {
        return prior.operation.promote(prior.operand.dtype());
    }
  
    auto forward() const -> decltype(auto) {     
        auto result = prior.operation.forward(prior.operand.forward()); 
        operation.forward(result, result);
        return result;
    }   
};
*/

#endif // EXPRESSIONS_HPP