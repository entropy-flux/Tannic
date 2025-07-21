// Copyright 2025 Eric Cardozo
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
#include "Indexing.hpp"  

namespace tannic {
class Tensor;
}

namespace tannic::expression { 

template<Expression Source, class... Indexes> 
static constexpr Shape shape(Source const& source, std::tuple<Indexes...> const& indexes) { 
    std::array<Shape::size_type, Shape::limit> sizes{};
    Shape::rank_type dimension = 0;
    Shape::rank_type rank = 0;
    const auto& shape = source.shape();

    auto process = [&](const auto& argument) {
        using Argument = std::decay_t<decltype(argument)>;
        if constexpr (std::is_same_v<Argument, indexing::Range>) { 
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


template<Expression Source, class... Indexes>
static constexpr Strides strides(Source const& source, std::tuple<Indexes...> const& indexes) {
    std::array<Strides::size_type, Strides::limit> sizes{};
    Strides::rank_type rank = 0;
    Strides::rank_type dimension = 0;
    const auto& strides = source.strides();

    auto process = [&](const auto& index) {
        using Argument = std::decay_t<decltype(index)>;
        if constexpr (std::is_same_v<Argument, indexing::Range>) { 
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

    while (dimension < source.rank()) {
        sizes[rank++] = strides[dimension++];
    }

    return Strides(sizes.begin(), sizes.begin() + rank);
} 


template<Expression Source, class... Indexes>
static constexpr auto offset(Source const& source, std::tuple<Indexes...> const& indexes) {
    std::ptrdiff_t result = 0;  
    std::ptrdiff_t dimension = 0; 
    const auto dsize = dsizeof(source.dtype()); 
    const auto& strides = source.strides();
    const auto& shape = source.shape();

    auto process = [&](const auto& argument) {
        using Argument = std::decay_t<decltype(argument)>;
        if constexpr (std::is_same_v<Argument, indexing::Range>) {  
            auto start = indexing::normalize(argument.start, shape[dimension]); 
            result += start * strides[dimension++] * dsize;
        } else if constexpr (std::is_integral_v<Argument>) { 
            auto index = indexing::normalize(argument, shape[dimension]);
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

template <Expression Source, class... Indexes>
class Slice {  
public:    
    constexpr Slice(typename Trait<Source>::Reference source, std::tuple<Indexes...> indexes)
    :   dtype_(source.dtype())
    ,   shape_(expression::shape(source, indexes))
    ,   strides_(expression::strides(source, indexes))
    ,   offset_(expression::offset(source, indexes))
    ,   source_(source)
    ,   indexes_(indexes)
    { 
    }    

    template<Integral Index>
    constexpr auto operator[](Index index) const { 
        return Slice<Source, Indexes..., Index>(source_, std::tuple_cat(indexes_, std::make_tuple(index)));
    } 

    constexpr auto operator[](indexing::Range range) const {    
        return Slice<Source, Indexes..., indexing::Range>(source_, std::tuple_cat(indexes_, std::make_tuple(range)));
    } 

    constexpr auto dtype() const {
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

    std::byte* buffer() {
        return source_.buffer() + offset_;
    }

    std::byte const* buffer() const {
        return source_.buffer() + offset_;
    }

    std::ptrdiff_t offset() const {
        return offset_ + source_.offset();
    }

    Tensor forward() const;

    template<typename T>
    void operator=(T value);

    template<typename T>
    bool operator==(T value) const;  

    void assign(std::byte const* value, std::ptrdiff_t offset) { 
        source_.assign(value, offset); 
    }

    bool compare(std::byte const* value, std::ptrdiff_t offset) const {
        return source_.compare(value, offset);
    }  

private: 
    type dtype_;
    Shape shape_;
    Strides strides_;
    std::ptrdiff_t offset_;
    typename Trait<Source>::Reference source_;              
    std::tuple<Indexes...> indexes_;    
};   

template<typename T>
static inline std::byte const* bytes(T const& reference) { 
    return reinterpret_cast<std::byte const*>(&reference);
}  

template <Expression Source, class... Indexes>
template <typename T>
void Slice<Source, Indexes...>::operator=(T value) {    
    auto copy = [this](std::byte const* value, std::ptrdiff_t offset) {
        if(rank() == 0) { 
            assign(value, offset);
            return;
        }

 
        std::vector<std::size_t> indexes(rank(), 0);
        bool done = false;  
        
        while (!done) {
            std::size_t position = offset;
            for (auto dimension = 0; dimension < rank(); ++dimension) {
                position += indexes[dimension] * strides_[dimension] * dsizeof(dtype_);
            } 
  
            assign(value, position); 
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
            copy(bytes(casted), offset()); 
            break;
        }
        case int16: {
            int16_t casted = value;
            copy(bytes(casted), offset()); 
            break;
        }
        case int32: {
            int32_t casted = value;
            copy(bytes(casted), offset());
            break;
        }
        case int64: {
            int64_t casted = value;
            copy(bytes(casted), offset());
            break;
        }
        case float32: {
            float casted = value;
            copy(bytes(casted), offset());
            break;
        }
        case float64: {
            double casted = value;
            copy(bytes(casted), offset());
            break;
        } 
        default:
            break;
        } 
}

template <Expression Source, class... Indexes>
template <typename T>
bool Slice<Source, Indexes...>::operator==(T value) const {    
    assert(rank() == 0 && "Cannot compare an scalar to a non scalar slice");
    switch (dtype_) {
        case int8: {
            int8_t casted = value;
            return compare(bytes(casted), offset()); 
        }
        case int16: {
            int16_t casted = value;
            return compare(bytes(casted), offset()); 
        }
        case int32: {
            int32_t casted = value;
            return compare(bytes(casted), offset());
        }
        case int64: {
            int64_t casted = value;
            return compare(bytes(casted), offset());
        }
        case float32: {
            float casted = value;
            return compare(bytes(casted), offset());
        }
        case float64: {
            double casted = value;
            return compare(bytes(casted), offset());
        } 
        default:
            return false;
        }
}  

} // namespace tannic::expression
 
#endif // SLICES_HPP 