#ifndef VIEWS_HPP
#define VIEWS_HPP

#include <utility> 
 
#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
#include "Concepts.hpp"

namespace tannic {

class Tensor;

namespace view {  

struct Range {
    int start;
    int stop;
};

template<Integral Index, Integral Size>
static constexpr inline Index normalize(Index index, Size bound) {
    if (index < 0) index += bound;
    assert(index >= 0 && index < bound && "Index out of bounds");
    return index;
}  

template<Integral Size>
static constexpr inline Range normalize(Range range, Size size) {
    int start = range.start < 0 ? size + range.start : range.start;
    int stop = range.stop < 0 ? size + range.stop + 1 : range.stop;
    return {start, stop};
}  
 
constexpr Shape transpose(Shape const& layout, std::pair<int, int> const& dimensions) { 
    auto rank = layout.rank();  
    std::array<Shape::size_type, Shape::limit> sizes{};
    for (Shape::rank_type dimension = 0; dimension < rank; ++dimension) {
        sizes[dimension] = layout[dimension];
    }

    std::swap(sizes[normalize(dimensions.first, rank)], sizes[normalize(dimensions.second, rank)]); 
    return Shape(sizes.begin(), sizes.begin() + rank);
}   
 
constexpr Strides transpose(Strides const& layout, std::pair<int, int> const& dimensions) { 
    auto rank = layout.rank();  
    std::array<Strides::size_type, Strides::limit> sizes{};
    for (Strides::rank_type dimension = 0; dimension < rank; ++dimension) {
        sizes[dimension] = layout[dimension];
    }

    std::swap(sizes[normalize(dimensions.first, rank)], sizes[normalize(dimensions.second, rank)]); 
    return Strides(sizes.begin(), sizes.begin() + rank);
}  

template<Operable Expression>
class Transpose {
public:
    typename Trait<Expression>::Reference expression; 
    std::pair<int, int> dimensions;

    constexpr Transpose(typename Trait<Expression>::Reference expression, std::pair<int, int> dimensions)
    :   expression(expression)
    ,   dimensions(dimensions)
    ,   shape_(transpose(expression.shape(), dimensions)) 
    ,   strides_(transpose(expression.strides(), dimensions))
    {}

    constexpr type dtype() const {
        return expression.dtype();
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    constexpr std::ptrdiff_t offset() const {
        return expression.offset();
    }

    Tensor forward() const;

private:
    Shape shape_; 
    Strides strides_;
};

template<Operable Expression>
constexpr auto transpose(Expression&& expression, int first, int second) {
    return Transpose<std::decay_t<Expression>>(
        std::forward<Expression>(expression),
        std::make_pair(first, second)
    );
}

} // namespace view

using view::transpose;

} // namespace tannic

#endif // VIEWS_HPP