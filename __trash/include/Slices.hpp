// Copyright 2025 Eric Cardozo
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

#ifndef SLICES_HPP
#define SLICES_HPP

#include <tuple>
#include <utility>
#include <cstddef>

#include "Types.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
 
class Tensor;

struct Range {
    int start = 0;
    int stop = -1;  
}; 

template<class Index, class Size>
static constexpr inline Index normalize(Index index, Size size) {
    return index < 0 ? size + index : index;
}  

template<class Size>
static constexpr inline Range normalize(Range range, Size size) {
    int start = range.start < 0 ? size + range.start : range.start;
    int stop = range.stop < 0 ? size + range.stop + 1 : range.stop;
    return {start, stop};
}  

template<class Expression, class... Indexes> 
static constexpr Shape shape(Expression const& expression, std::tuple<Indexes...> const& indexes) { 
    std::array<Shape::size_type, Shape::limit> sizes{};
    Shape::rank_type dimension = 0;
    Shape::rank_type rank = 0;
    const auto& shape = expression.shape();

    auto process = [&](const auto& argument) {
        using Argument = std::decay_t<decltype(argument)>;
        if constexpr (std::is_same_v<Argument, Range>) { 
            auto range = normalize(argument, shape[dimension]);
            auto size = range.stop - range.start;
            assert(size >= 0);
            sizes[rank++] = size;
            dimension++;
        } else if constexpr (std::is_integral_v<Argument>) {
            dimension++;
        } else {
            static_assert(sizeof(Argument) == 0, "Unsupported index type in Slice");
        }
    };

    std::apply([&](const auto&... arguments) {
        (process(arguments), ...);
    }, indexes);

    while (dimension < shape.rank()) {
        sizes[rank++] = shape[dimension++];
    }

    return Shape(sizes.begin(), sizes.begin() + rank);
}


template<class Expression, class... Indexes>
static constexpr Strides strides(Expression const& expression, std::tuple<Indexes...> const& indexes) {
    std::array<Strides::size_type, Strides::limit> sizes{};
    Strides::rank_type rank = 0;
    Strides::rank_type dimension = 0;
    const auto& strides = expression.strides();

    auto process = [&](const auto& index) {
        using Argument = std::decay_t<decltype(index)>;
        if constexpr (std::is_same_v<Argument, Range>) { 
            sizes[rank++] = strides[dimension++];
        } else if constexpr (std::is_integral_v<Argument>) { 
            ++dimension;
        } else {
            static_assert(sizeof(Argument) == 0, "Unsupported index type in Slice strides");
        }
    };

    std::apply([&](const auto&... indices) {
        (process(indices), ...);
    }, indexes);

    while (dimension < expression.rank()) {
        sizes[rank++] = strides[dimension++];
    }

    return Strides(sizes.begin(), sizes.begin() + rank);
}


template<class Expression, class... Indexes>
static constexpr auto offset(Expression const& expression, std::tuple<Indexes...> const& indexes) {
    std::ptrdiff_t result = 0;
    std::ptrdiff_t dimension = 0; 
    const auto& strides = expression.strides();
    const auto& shape = expression.shape();

    auto process = [&](const auto& argument) {
        using Argument = std::decay_t<decltype(argument)>;
        if constexpr (std::is_same_v<Argument, Range>) {  
            auto start = normalize(argument.start, shape[dimension]); 
            result += start * strides[dimension++];
        } else if constexpr (std::is_integral_v<Argument>) { 
            auto index = normalize(argument, shape[dimension]);
            result += index * strides[dimension++];
        } else {
            static_assert(sizeof(Argument) == 0, "Unsupported index type in Slice offset");
        }
    };

    std::apply([&](const auto&... indices) {
        (process(indices), ...);
    }, indexes);
    return result * dsizeof(expression.dtype());
} 





namespace expression {

template <class Expression, class... Indexes>
class Slice {  
public:   
    typename Trait<Expression>::Reference expression;              
    std::tuple<Indexes...> indexes;    

    constexpr Slice(typename Trait<Expression>::Reference expression, std::tuple<Indexes...> indexes)
    :   expression(expression)
    ,   indexes(indexes)
    ,   dtype_(expression.dtype())
    ,   shape_(::shape(expression, indexes))
    ,   strides_(::strides(expression, indexes))
    ,   offset_(::offset(expression, indexes))
    {}    

    template<class Index>
    constexpr auto operator[](Index index) const { 
        return Slice<Expression, Indexes..., Index>(expression, std::tuple_cat(indexes, std::make_tuple(index)));
    } 

    constexpr auto operator[](Range range) const {    
        return Slice<Expression, Indexes..., Range>(expression, std::tuple_cat(indexes, std::make_tuple(range)));
    } 

    constexpr type dtype() const {
        return dtype_;
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    constexpr std::ptrdiff_t offset() const {
        return offset_;
    }
 
    std::byte* address() {
        return expression.address();
    }

    std::byte const* address() const {
        return expression.address();
    }

    Tensor forward() const;

    template<typename T>
    void operator=(T value);

    template<typename T>
    bool operator==(T value) const;

    void assign(std::byte const* value, std::ptrdiff_t offset) {
        expression.assign(value, offset);
    }

    bool compare(std::byte const* value, std::ptrdiff_t offset) const {
        return expression.compare(value, offset);
    }

private: 
    type dtype_;
    Shape shape_;
    Strides strides_;
    std::ptrdiff_t offset_;
}; 

} // namespace expression
 

#endif // SLICES_HPP