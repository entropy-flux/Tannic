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

#ifndef SHAPE_HPP
#define SHAPE_HPP
  
#include <type_traits> 
#include <array>
#include <cstdint>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <numeric>

template<typename T>
concept Iterable = requires(T type) {
    std::begin(type);
    std::end(type);
};

class Shape {
public:
    using index_type = int8_t;
    using rank_type = uint8_t;
    using size_type = std::size_t;
    static constexpr uint8_t limit = 8;  

    constexpr Shape() noexcept = default;

    template<typename... Sizes>
    constexpr explicit Shape(Sizes... sizes) 
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {      
        if (sizeof...(sizes) > limit) {
            throw "Rank limit exceeded";
        }
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

    template<Iterable Sizes>
    constexpr explicit Shape(Sizes&& sizes) {
        size_type dimension = 0;
        for (auto size : sizes) {
            sizes_[dimension++] = static_cast<size_type>(size);
        }
        
        if (dimension >= limit) {
            throw "Rank limit exceeded";
        }
        rank_ = dimension;
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    } 


    template<std::input_iterator Iterator>
    constexpr Shape(Iterator begin, Iterator end) {
        size_type dimension = 0;
        for (Iterator iterator = begin; iterator != end; ++iterator) {
            if (dimension >= limit) {
                throw "Rank limit exceeded";
            }
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension;
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

    constexpr rank_type rank() const noexcept { return rank_; }
    constexpr size_type operator[](index_type dimension) const noexcept { return sizes_[dimension]; }
    constexpr size_type& operator[](index_type dimension) noexcept { return sizes_[dimension]; }
    constexpr size_type size() const noexcept { return size_; }
    
    constexpr auto begin() { return sizes_.begin(); }
    constexpr auto end() { return sizes_.begin() + rank_; }

    constexpr auto begin() const { return sizes_.begin(); }
    constexpr auto end() const { return sizes_.begin() + rank_; }

    constexpr auto cbegin() const { return sizes_.cbegin(); }
    constexpr auto cend() const { return sizes_.cbegin() + rank_; }

    constexpr auto front() const { return sizes_.front(); }
    constexpr auto back() const { 
        assert(rank_ > 0 && "Cannot call back() on an empty Shape");
        return sizes_[rank_ - 1];
    }    
        
    constexpr rank_type normalize(index_type index, rank_type extra = 0) const { 
        rank_type bound = rank() + extra;
        if (index < 0) index += bound;
        assert(index >= 0  && index < bound && "Index out of bound");
        return static_cast<rank_type>(index);
    }

    constexpr Shape transpose(index_type first, index_type second) const { 
        Shape result = *this;
        std::swap(result.sizes_[normalize(first)], result.sizes_[normalize(second)]);
        return result;
    }


        
private:
    size_type size_;
    std::array<size_type, limit> sizes_{};
    size_type rank_{0};
};
 
constexpr bool operator==(const Shape& first, const Shape& second) {
    if (first.rank() != second.rank()) return false;
    for (Shape::size_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
} 

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "Shape(";
    for (Shape::rank_type dimension = 0; dimension < shape.rank(); ++dimension) {
        os << shape[dimension];
        if (dimension + 1 < shape.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

#endif // SHAPE_HPP