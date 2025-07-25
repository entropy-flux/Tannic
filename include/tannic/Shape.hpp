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


#ifndef SHAPE_HPP
#define SHAPE_HPP
  
#include <type_traits> 
#include <array>
#include <cstdint>
#include <cassert> 
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <ostream>
 
#include "Concepts.hpp" 
#include "Indexing.hpp"

namespace tannic {
    
class Shape {
public:
    static constexpr uint8_t limit = 8;  

public:
    using rank_type = uint8_t;
    using size_type = std::size_t; 
 
    constexpr Shape() noexcept = default;

    constexpr Shape(std::initializer_list<size_type> shape) {
        assert(shape.size() < limit && "Rank limit exceeded");
        rank_ = static_cast<size_type>(shape.size());
        size_type dimension = 0;
        for (auto size : shape) {
            sizes_[dimension++] = size;
        }
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>());
    }

    template<typename... Sizes>
    constexpr explicit Shape(Sizes... sizes) 
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {      
        assert(rank_ < limit && "Rank limit exceeded");
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

    template<Iterable Sizes>
    constexpr explicit Shape(Sizes&& sizes) {
        size_type dimension = 0;
        for (auto size : sizes) {
            sizes_[dimension++] = static_cast<size_type>(size);
        }
        rank_ = dimension;
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    } 

    template<Iterator Iterator>
    constexpr Shape(Iterator begin, Iterator end) {
        size_type dimension = 0;
        for (Iterator iterator = begin; iterator != end; ++iterator) { 
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension; 
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

public:
    constexpr size_type* address() noexcept {
        return sizes_.data();
    }

    constexpr size_type const* address() const noexcept {
        return sizes_.data();
    }

    constexpr rank_type rank() const noexcept { 
        return rank_; 
    }
    
    constexpr size_type size() const noexcept { 
        return size_; 
    }
    
    constexpr auto begin() { 
        return sizes_.begin(); 
    }
    
    constexpr auto end() { 
        return sizes_.begin() + rank_; 
    }

    constexpr auto begin() const { 
        return sizes_.begin(); 
    }
    
    constexpr auto end() const { 
        return sizes_.begin() + rank_; 
    }

    constexpr auto cbegin() const { 
        return sizes_.cbegin(); 
    }
    
    constexpr auto cend() const { 
        return sizes_.cbegin() + rank_; 
    }

    constexpr auto front() const noexcept { 
        return sizes_.front(); 
    }

    constexpr auto back() const { 
        assert(rank_ > 0 && "Cannot call back() on an empty Shape");
        return sizes_[rank_ - 1];
    }       

    template<Integral Index>
    constexpr auto const& operator[](Index index) const { 
        return sizes_[indexing::normalize(index, rank())]; 
    }

    template<Integral Index>
    constexpr auto& operator[](Index index) {
        return sizes_[indexing::normalize(index, rank())]; 
    } 


private:
    rank_type rank_{0};
    size_type size_{1}; 
    std::array<size_type, limit> sizes_{}; 
};
 
constexpr bool operator==(Shape const& first, Shape const& second) {
    if (first.rank() != second.rank()) return false;
    for (Shape::size_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
}  

inline std::ostream& operator<<(std::ostream& os, Shape const& shape) {
    os << "Shape(";
    for (Shape::rank_type dimension = 0; dimension < shape.rank(); ++dimension) {
        os << static_cast<unsigned int>(shape[dimension]);
        if (dimension + 1 < shape.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
} 

} // namespace tannic

#endif // SHAPE_HPP