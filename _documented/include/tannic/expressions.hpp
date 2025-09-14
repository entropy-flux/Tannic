#ifndef EXPRESSION_HPP
#define EXPRESSION_HPP

#include <tuple>
#include "traits.hpp"
#include "concepts.hpp"

namespace tannic::expression {

template<class Operation, Composable ... Operands>
class Expression {
    public:
    Operation operation;
    std::tuple<typename Trait<Operands>::Reference...> operands;

    constexpr Expression(Operation operation, typename Trait<Operands>::Reference ... operands) 
    :   operation(std::move(operation))
    ,   operands(std::make_tuple(operands...)) {}
};

} 

#endif // EXPRESSION_HPP