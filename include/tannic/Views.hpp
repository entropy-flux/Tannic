#ifndef VIEWS_HPP
#define VIEWS_HPP

#include <utility> 
#include <algorithm>
#include <numeric>
#include <vector>

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
        std::size_t elements = 0;
        for (std::size_t dimension = 0; dimension < sizeof...(indexes); ++dimension) {
            elements += strides_[dimension] * (shape_[dimension] - 1);
        } 
        elements += 1;  
        assert(elements == std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{})
        && "Shape mismatch: view must preserve total number of elements");
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
 
 
template<Expression Source>
class Transpose {
public: 
    constexpr Transpose(typename Trait<Source>::Reference source, std::pair<int, int> dimensions)
    :   shape_(source.shape()) 
    ,   strides_(source.strides())
    ,   source_(source)
    ,   dimensions_(dimensions) {  
        auto rank = source.shape().rank();    
        std::swap(shape_[indexing::normalize(dimensions.first, rank)], shape_[indexing::normalize(dimensions.second, rank)]); 
        std::swap(strides_[indexing::normalize(dimensions.first, rank)], strides_[indexing::normalize(dimensions.second, rank)]);   
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