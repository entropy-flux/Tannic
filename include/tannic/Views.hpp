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

namespace expression {   

template<Expression Source>
class Reshape {
public: 
    template<Integral... Indexes>
    constexpr Reshape(Trait<Source>::Reference source, Indexes... indexes)
    :   shape_(indexes...) 
    ,   strides_(shape_)
    ,   source_(source) {
        assert(source_.shape().size() == shape().size() && "Shape mismatch: view must preserve total number of elements");
    }

    constexpr type dtype() const {
        return source_.dtype();
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    std::ptrdiff_t offset() const {
        return source_.offset();
    }
 
    Tensor forward() const;

private:
    Shape shape_;
    Strides strides_;
    typename Trait<Source>::Reference source_;                
}; 
 
constexpr Shape transpose(Shape const& layout, std::pair<int, int> const& dimensions) { 
    auto rank = layout.rank();  
    std::array<Shape::size_type, Shape::limit> sizes{};
    for (Shape::rank_type dimension = 0; dimension < rank; ++dimension) {
        sizes[dimension] = layout[dimension];
    }

    std::swap(sizes[indexing::normalize(dimensions.first, rank)], sizes[indexing::normalize(dimensions.second, rank)]); 
    return Shape(sizes.begin(), sizes.begin() + rank);
}   
 
constexpr Strides transpose(Strides const& layout, std::pair<int, int> const& dimensions) { 
    auto rank = layout.rank();  
    std::array<Strides::size_type, Strides::limit> sizes{};
    for (Strides::rank_type dimension = 0; dimension < rank; ++dimension) {
        sizes[dimension] = layout[dimension];
    }

    std::swap(sizes[indexing::normalize(dimensions.first, rank)], sizes[indexing::normalize(dimensions.second, rank)]); 
    return Strides(sizes.begin(), sizes.begin() + rank);
}  


/*----------------------------------------------------------------*/
/*TODO: This is an special case of Permutation and may get it's own file. however the api will remain the same. */
template<Expression Source>
class Transpose {
public: 
    constexpr Transpose(typename Trait<Source>::Reference source, std::pair<int, int> dimensions)
    :   shape_(transpose(source.shape(), dimensions)) 
    ,   strides_(transpose(source.strides(), dimensions))
    ,   source_(source)
    ,   dimensions_(dimensions)
    {}

    constexpr type dtype() const {
        return source_.dtype();
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    } 

    std::ptrdiff_t offset() const {
        return source_.offset();
    }

    Tensor forward() const;

private:
    Shape shape_; 
    Strides strides_;
    typename Trait<Source>::Reference source_; 
    std::pair<int, int> dimensions_;
};

template<Expression Source, Integral ... Indexes>
constexpr auto view(Source&& source, Indexes ... indexes) {
    return Reshape<Source>(
        std::forward<Source>(source), indexes...
    );
}

template<Expression Source>
constexpr auto transpose(Source&& source, int first, int second) {
    return Transpose<Source>(
        std::forward<Source>(source),
        std::make_pair(first, second)
    );
} 

/*----------------------------------------------------------------*/ 

} // namespace expression

using expression::view;
using expression::transpose;

} // namespace tannic

#endif // VIEWS_HPP