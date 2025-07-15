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
#include <vector>

#include "Types.hpp" 
#include "Traits.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
#include "Views.hpp"
 
namespace tannic {

class Tensor; 
 
namespace view {  

template<Operable Expression, class... Indexes> 
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


template<Operable Expression, class... Indexes>
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


template<Operable Expression, class... Indexes>
static constexpr auto offset(Expression const& expression, std::tuple<Indexes...> const& indexes) {
    std::ptrdiff_t result = expression.offset();  
    std::ptrdiff_t dimension = 0; 
    const auto dsize = dsizeof(expression.dtype()); 
    const auto& strides = expression.strides();
    const auto& shape = expression.shape();

    auto process = [&](const auto& argument) {
        using Argument = std::decay_t<decltype(argument)>;
        if constexpr (std::is_same_v<Argument, Range>) {  
            auto start = normalize(argument.start, shape[dimension]); 
            result += start * strides[dimension++] * dsize;
        } else if constexpr (std::is_integral_v<Argument>) { 
            auto index = normalize(argument, shape[dimension]);
            result += index * strides[dimension++] * dsize;
        } else {
            static_assert(sizeof(Argument) == 0, "Unsupported index type in Slice offset");
        }
    };

    std::apply([&](const auto&... indices) {
        (process(indices), ...);
    }, indexes);
    return result;
} 

template <Operable Expression, class... Indexes>
class Slice {  
public:   
    typename Trait<Expression>::Reference expression;              
    std::tuple<Indexes...> indexes;    

    constexpr Slice(typename Trait<Expression>::Reference expression, std::tuple<Indexes...> indexes)
    :   expression(expression)
    ,   indexes(indexes)
    ,   dtype_(expression.dtype())
    ,   shape_(view::shape(expression, indexes))
    ,   strides_(view::strides(expression, indexes))
    ,   offset_(view::offset(expression, indexes))
    {}    

    template<Integral Index>
    constexpr auto operator[](Index index) const { 
        return Slice<Expression, Indexes..., Index>(expression, std::tuple_cat(indexes, std::make_tuple(index)));
    } 

    constexpr auto operator[](Range range) const {    
        return Slice<Expression, Indexes..., Range>(expression, std::tuple_cat(indexes, std::make_tuple(range)));
    } 

    constexpr type dtype() const {
        return dtype_;
    }

    constexpr auto rank() const {
        return shape_.rank();
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

    std::byte* buffer() {
        return expression.buffer() + offset_;
    }

    std::byte const* buffer() const {
        return expression.buffer() + offset_;
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

protected:

private: 
    type dtype_;
    Shape shape_;
    Strides strides_;
    std::ptrdiff_t offset_;
};  
 

template<typename T>
static inline std::byte const* bytes(T const& reference) { 
    return reinterpret_cast<std::byte const*>(&reference);
}  

template <Operable Expression, class... Indexes>
template <typename T>
void Slice<Expression, Indexes...>::operator=(T value) {   
    auto copy = [this](std::byte const* value, std::ptrdiff_t offset) {
        if(rank() == 0) {
            expression.assign(value, offset);
            return;
        }

        std::vector<std::size_t> indexes(rank(), 0);
        bool done = false;  
        
        while (!done) {
            std::size_t position = offset;
            for (auto dimension = 0; dimension < rank(); ++dimension) {
                position += indexes[dimension] * strides_[dimension];
            }
            
            expression.assign(value, position); 
            done = true;
            for (int dimension = rank() - 1; dimension >= 0; --dimension) {
                if (++indexes[dimension] < shape_[dimension]) {
                    done = false;
                    break;
                }
                indexes[dimension] = 0;
            }
        }
    };

    switch (dtype_) {
        case int8: {
            int8_t casted = value;
            copy(bytes(casted), offset_); 
            break;
        }
        case int16: {
            int16_t casted = value;
            copy(bytes(casted), offset_); 
            break;
        }
        case int32: {
            int32_t casted = value;
            copy(bytes(casted), offset_);
            break;
        }
        case int64: {
            int64_t casted = value;
            copy(bytes(casted), offset_);
            break;
        }
        case float32: {
            float casted = value;
            copy(bytes(casted), offset_);
            break;
        }
        case float64: {
            double casted = value;
            copy(bytes(casted), offset_);
            break;
        } 
        default:
            break;
        }

}

template <Operable Expression, class... Indexes>
template <typename T>
bool Slice<Expression, Indexes...>::operator==(T value) const {    
    assert(rank() == 0 && "Cannot compare an scalar to a non scalar slice");
    switch (dtype_) {
        case int8: {
            int8_t casted = value;
            return expression.compare(bytes(casted), offset_); 
        }
        case int16: {
            int16_t casted = value;
            return expression.compare(bytes(casted), offset_); 
        }
        case int32: {
            int32_t casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case int64: {
            int64_t casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case float32: {
            float casted = value;
            return expression.compare(bytes(casted), offset_);
        }
        case float64: {
            double casted = value;
            return expression.compare(bytes(casted), offset_);
        } 
        default:
            return false;
        }
}  
 
} // namespace view

using range = view::Range;

} // namespace tannic

#endif // SLICES_HPP