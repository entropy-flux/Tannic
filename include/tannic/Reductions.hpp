#ifndef REDUCTIONS_HPP
#define REDUCTIONS_HPP 
 
#include <array>
#include <vector>
#include <cassert>

#include "Concepts.hpp"
#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp" 
#include "Tensor.hpp" 

namespace tannic {
    
namespace expression {

template<class Reducer, Expression Operand>
class Reduction {
public:
    Reducer reducer;
    typename Trait<Operand>::Reference operand;

    constexpr Reduction(Reducer reducer, typename Trait<Operand>::Reference operand)
    :   reducer(reducer)
    ,   operand(operand)
    ,   dtype_(reducer.reduce(operand.dtype()))
    ,   shape_(reducer.reduce(operand.shape()))
    ,   strides_(reducer.reduce(operand.strides())) {}

    constexpr type dtype() const {
        return dtype_;
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    std::ptrdiff_t offset() const {
        return 0;
    }

    Tensor forward() const {   
        Tensor source = source.forward();
        Tensor result(dtype_, shape_, strides_, offset());          
        reducer.forward(source, result);
        return result;
    }

private:
    type dtype_;
    Shape shape_;
    Strides strides_;

}; 

struct Argmax {   
    void forward(Tensor const&, Tensor&) const;
    
    constexpr type reduce(type dtype) const {
        return int64;
    }

    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1 && "Argmax requires at least 1D tensor");
        return Shape(shape.begin(), shape.end()-1);
    }

    constexpr Strides reduce(Strides const& strides) const {
        uint8_t rank = strides.rank() - 1; 
        std::vector<std::size_t> sizes(rank);
        for(uint8_t dimension = 0; dimension < rank - 2; ++dimension) {
            sizes[dimension] = strides[dimension];
        }
        sizes[rank - 1] = strides.back();
        return Strides(sizes.begin(), sizes.end());
    }
};

struct Argmin {   
    void forward(Tensor const&, Tensor&) const; 

    constexpr type reduce(type dtype) const {
        return int64;
    }

    constexpr Shape reduce(Shape const& shape) const {
        assert(shape.rank() >= 1 && "Argmax requires at least 1D tensor");
        return Shape(shape.begin(), shape.end()-1);
    }

    constexpr Strides reduce(Strides const& strides) const {
        uint8_t rank = strides.rank() - 1; 
        std::vector<std::size_t> sizes(rank);
        for(uint8_t dimension = 0; dimension < rank - 2; ++dimension) {
            sizes[dimension] = strides[dimension];
        }
        sizes[rank - 1] = strides.back();
        return Strides(sizes.begin(), sizes.end());
    }
}; 
 
template<Expression Source>
constexpr auto argmax(Source&& source) {
    return Reduction<Argmax, Source>{
        {}, std::forward<Source>(source) 
    };
}

template<Expression Source>
constexpr auto argmin(Source&& source) {
    return Reduction<Argmin, Source>{
        {}, std::forward<Source>(source) 
    };
}
 
} // namespace expression

using expression::argmax;
using expression::argmin;

} // namespace tannic

#endif