// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Tensor Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
 
#ifndef VIEWS_HPP
#define VIEWS_HPP  

#include <utility> 
#include <algorithm>
#include <numeric>
#include <vector>

#include "types.hpp"
#include "traits.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "concepts.hpp"
#include "exceptions.hpp"

namespace tannic {

class Tensor;

} namespace tannic::expression {    


template<Composable Source>
class View {
    // tag to be used later in derivatives
};
    
template<Composable Source>
class Reshape : public View<Source> {
public:    
    template<Integral... Sizes>  
    constexpr Reshape(typename Trait<Source>::Reference source, Sizes... sizes)
    :   source_(source)
    { 
        std::array<long long, sizeof...(Sizes)> requested{ static_cast<long long>(sizes)... };
  
        std::size_t nelements = 1;
        std::size_t stride = 1;
        for (auto dimension = 0; dimension < source.shape().rank(); ++dimension) {
            auto index = source.shape().rank() - 1 - dimension; 
            if (source.strides()[index] != stride) {
                throw Exception("Only contiguous tensors allowed in view");
            }
            nelements *= source.shape()[dimension];
            stride *= source.shape()[index];
        }
        
        int inferred = -1;
        std::size_t accumulated = 1; 
        for (auto dimension = 0; dimension < requested.size(); ++dimension) { 
            auto index = requested.size() - 1 - dimension; 

            if (requested[dimension] == -1) {
                if (inferred != -1) throw Exception("Only one dimension can be inferred (-1) in view");
                inferred = dimension;
            } 
            
            else if (requested[dimension] < 0) {
                throw Exception("Invalid negative dimension in view");
            } 
            
            else {
                accumulated *= requested[dimension];
            }
            shape_.expand(requested[dimension]);
            strides_.expand(1);
        }
 
        if (inferred != -1) {
            if (nelements % accumulated != 0) throw Exception("Cannot infer dimension: source elements not divisible");
            shape_[inferred] = nelements / accumulated; 
        } 

        else if (accumulated != nelements) {
            throw Exception("Shape mismatch: view must preserve total number of elements");
        }

        for (auto dimension = shape_.rank() - 2; dimension >= 0; --dimension) {
            strides_[dimension] = strides_[dimension + 1] * shape_[dimension + 1];
        } 
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
 
    Tensor forward(Context const& context) const;

private:
    Shape shape_;
    Strides strides_;
    typename Trait<Source>::Reference source_;                
};   


template<Composable Source>
class Transpose : public View<Source> {
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

    Tensor forward(Context const& context) const;

private:
    Shape shape_; 
    Strides strides_;
    typename Trait<Source>::Reference source_; 
    std::pair<int, int> dimensions_;
};   

template<Composable Source>
class Permutation : public View<Source> {
public:

    template<Integral... Indexes>
    constexpr Permutation(typename Trait<Source>::Reference source, Indexes... indexes)
        : source_(source)
    {
        if (sizeof...(Indexes) != source_.shape().rank()) {
            throw Exception("Permutation rank must equal tensor rank");
        }

        (([&]{
            int dimension = indexing::normalize(indexes, source_.shape().rank());
            shape_.expand(source_.shape()[dimension]);
            strides_.expand(source_.strides()[dimension]);
        }()), ...);
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

    Tensor forward(Context const& context) const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;              
};



template<Composable Source>
class Expansion : public View<Source> {
public:
    template<Integral... Sizes>
    constexpr Expansion(typename Trait<Source>::Reference source, Sizes... sizes)
    :   source_(source) {
        std::array<long long, sizeof...(Sizes)> requested{ static_cast<long long>(sizes)... }; 
 
        if (requested.size() < source.shape().rank()) 
            throw Exception("Expansion target rank must be >= source rank");

        std::size_t offset = requested.size() - source.shape().rank();
        for (std::size_t dimension = 0; dimension < requested.size(); ++dimension) {
            long long index = requested[dimension];
            std::size_t target;

            if (index == -1) {
                if (dimension < offset) {
                    throw Exception("Cannot use -1 for new leading dimensions");
                }
                target = source.shape()[dimension - offset];
            } else if (index <= 0) {
                throw Exception("Expansion size must be positive or -1");
            } else {
                target = static_cast<std::size_t>(index);
            }
 
            if (dimension < offset) { 
                shape_.expand(target);
                strides_.expand(0);
            } 
            
            else { 
                if (source.shape()[dimension - offset] == 1 && target > 1) {
                    shape_.expand(target);
                    strides_.expand(0);  // broadcast
                } else if (source.shape()[dimension - offset] == target) {
                    shape_.expand(target);
                    strides_.expand(source.strides()[dimension - offset]);
                } else {
                    throw Exception("Expansion only allows -1 (keep) or broadcasting singleton dims");
                }
            }
        }
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

    Tensor forward(Context const& context) const;

private:
    Shape shape_;
    Strides strides_;
    typename Trait<Source>::Reference source_;
};

template<Composable Source>
class Squeeze : public View<Source> {
public:
    constexpr Squeeze(typename Trait<Source>::Reference source)
    : source_(source) {
        for (auto dimension = 0; dimension < source.shape().rank(); ++dimension) {
            if (source.shape()[dimension] != 1) {
                shape_.expand(source.shape()[dimension]);
                strides_.expand(source.strides()[dimension]);
            }
        }
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

    Tensor forward(Context const& context) const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;
};


template<Composable Source>
class Unsqueeze : public View<Source> {
public:

    template<Integral... Axes>
    constexpr Unsqueeze(typename Trait<Source>::Reference source, Axes... axes)
    : source_(source) {
        auto rank = source.shape().rank();
        std::vector<std::size_t> normalized{ static_cast<std::size_t>(indexing::normalize(axes, rank + sizeof...(axes)))... };
        std::sort(normalized.begin(), normalized.end());

        size_t dimensions = rank + normalized.size();
        size_t index = 0;
        size_t axis = 0;

        for (auto dimension = 0; dimension < dimensions; ++dimension) {
            if (axis < normalized.size() && dimension == normalized[axis]) {
                shape_.expand(1);
                strides_.expand( (index < source.strides().rank()) ? source.strides()[index] : 1 );
                ++axis;
            } else {
                shape_.expand(source.shape()[index]);
                strides_.expand(source.strides()[index]);
                ++index;
            }
        }
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

    Tensor forward(Context const& context) const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;
};


template<Composable Source>
class Flatten : public View<Source> {
public:
    constexpr Flatten(typename Trait<Source>::Reference source, int start = 0, int end = -1)
    :   source_(source) {
        auto rank = source.shape().rank();
 
        start = indexing::normalize(start, rank);
        end   = indexing::normalize(end, rank);

        if (start > end) {
            throw Exception("Flatten requires start_dim <= end_dim");
        }
 
        for (int dimension = 0; dimension < start; ++dimension) {
            shape_.expand(source.shape()[dimension]);
            strides_.expand(source.strides()[dimension]);
        }
 
        std::size_t flattened = 1;
        for (int dimension = start; dimension <= end; ++dimension) {
            flattened *= source.shape()[dimension];
        }
        shape_.expand(flattened);
        strides_.expand(source.strides()[end]);  
 
        for (int dimension = end + 1; dimension < rank; ++dimension) {
            shape_.expand(source.shape()[dimension]);
            strides_.expand(source.strides()[dimension]);
        }
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

    std::ptrdiff_t offset() const { return source_.offset(); }

    Tensor forward(Context const& context) const;

private:
    Shape shape_{};
    Strides strides_{};
    typename Trait<Source>::Reference source_;
};

 

  
/*
----------------------------------------------------------------------------------------------------
*/


/**
 * @brief Creates a reshaped view of a tensor or expression.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Indexes New shape dimensions (integral values).
 * @param source The source expression.
 * @param indexes Dimension sizes for the new shape.
 * @return A `View` view expression.
 */
template<Composable Source, Integral ... Indexes>
constexpr auto view(Source&& source, Indexes ... indexes) {
    return Reshape<Source>(
        std::forward<Source>(source), indexes...
    );
} 


/**
 * @brief Creates a transposed view of a tensor or expression by swapping two dimensions.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source expression.
 * @param first First dimension index to swap.
 * @param second Second dimension index to swap.
 * @return A `Transpose` view expression.
 */
template<Composable Source>
constexpr auto transpose(Source&& source, int first, int second) {
    return Transpose<Source>(
        std::forward<Source>(source),
        std::make_pair(first, second)
    );
} 

/**
 * @brief Creates a permuted view of a tensor or expression.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Indexes Integral indices specifying the permutation order.
 * @param source The source expression.
 * @param indexes Sequence of dimension indices indicating the new axis order.
 * @return A `Permutation` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = permute(X, 2, 0, 1); // shape becomes (4, 2, 3)
 * ```
 */
template<Composable Source, Integral ... Indexes>
constexpr auto permute(Source&& source, Indexes... indexes) {
    return Permutation<Source>(
        std::forward<Source>(source), indexes...
    );
}

  
/**
 * @brief Creates an expanded view of a tensor, broadcasting singleton dimensions.
 *
 * This function returns an `Expansion` expression that allows a tensor to be 
 * “expanded” along dimensions of size 1 without copying data. Expansion is only 
 * allowed along singleton dimensions; other dimensions must match the requested size.
 *
 * @tparam Source The tensor or expression type to expand.
 * @tparam Sizes Integral dimension sizes for the expanded view.
 * @param source The source tensor or expression.
 * @param sizes The target shape for the expanded view.
 * @return An `Expansion` object representing the broadcasted view.
 *
 * @throws Exception if:
 *   - The number of dimensions does not match the source rank.
 *   - A non-singleton dimension in the source does not match the requested size.
 *
 * Example usage:
 * ```cpp
 * Tensor X(float32, {1, 3}); // shape: (1, 3)
 * auto Y = expand(X, 4, 3);  // shape: (4, 3), broadcasts along the first dimension
 *
 * std::cout << Y.shape() << std::endl;   // prints (4, 3)
 * std::cout << Y.strides() << std::endl; // prints (0, original_stride[1])
 * ```
 */
template<Composable Source, Integral... Sizes>
constexpr auto expand(Source&& source, Sizes... sizes) {
    return Expansion<Source>(std::forward<Source>(source), sizes...);
}


/**
 * @brief Removes all singleton dimensions from a tensor (squeeze).
 *
 * This function returns a `Squeeze` expression that reinterprets the
 * source tensor without its size-1 dimensions.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source tensor or expression.
 * @return A `Squeeze` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {1, 3, 1});
 * auto Y = squeeze(X); // shape: (3)
 * ```
 */
template<Composable Source>
constexpr auto squeeze(Source&& source) {
    return Squeeze<Source>(std::forward<Source>(source));
}


/**
 * @brief Inserts singleton dimensions at the specified axes (unsqueeze).
 *
 * This function returns an `Unsqueeze` expression that reinterprets
 * the source tensor with new dimensions of size 1 added.
 *
 * @tparam Source The expression or tensor type.
 * @tparam Axes One or more integral indices where new dimensions will be inserted.
 * @param source The source tensor or expression.
 * @param axes Axis indices (negative indices allowed).
 * @return An `Unsqueeze` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {3});
 * auto Y = unsqueeze(X, 0);  // shape: (1, 3)
 * auto Z = unsqueeze(X, -1); // shape: (3, 1)
 * ```
 */
template<Composable Source, Integral... Axes>
constexpr auto unsqueeze(Source&& source, Axes... axes) {
    return Unsqueeze<Source>(std::forward<Source>(source), axes...);
}


/**
 * @brief Flattens dimensions of a tensor into a single dimension.
 *
 * @tparam Source The expression or tensor type.
 * @param source The source tensor or expression.
 * @param start_dim First dimension to flatten (default = 0).
 * @param end_dim Last dimension to flatten (default = -1, meaning last dim).
 * @return A `Flatten` view expression.
 *
 * Example:
 * ```cpp
 * Tensor X(float32, {2, 3, 4});
 * auto Y = flatten(X, 1, -1); // shape: (2, 12)
 * auto Z = flatten(X);        // shape: (24)
 * ```
 */
template<Composable Source>
constexpr auto flatten(Source&& source, int start = 0, int end   = -1) {
    return Flatten<Source>(std::forward<Source>(source), start, end);
}


} namespace tannic {

using expression::view;
using expression::transpose;
using expression::permute;
using expression::expand;
using expression::squeeze;
using expression::unsqueeze;
using expression::flatten;

} // namespace tannic

#endif // VIEWS_HPP