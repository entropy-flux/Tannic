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

#ifndef STRIDES_HPP
#define STRIDES_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <ostream>

#include "Shape.hpp"

class Strides {
public:
    static constexpr uint8_t limit = 8;  

public:
    using size_type = std::size_t;
    using rank_type = Shape::rank_type;

    constexpr Strides() noexcept = default;
 
    template<typename... Sizes>
    constexpr explicit Strides(Sizes... sizes)
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {
        assert(rank_ < limit && "Strides rank limit exceeded");
    }
  
    template<Iterator Iterator>
    constexpr Strides(Iterator begin, Iterator end) { 
        size_type dimension = 0;
        for (auto iterator = begin; iterator != end; ++iterator) {
            assert(dimension < limit && "Strides rank limit exceeded");
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension;
    }
 
    constexpr explicit Strides(const Shape& shape) { 
        rank_ = shape.rank();
        if (rank_ == 0) return;
        
        sizes_[rank_ - 1] = 1;
        for (int size = rank_ - 2; size >= 0; --size) {
            sizes_[size] = sizes_[size + 1] * shape[size + 1];
        }
    }

    constexpr size_type* address() noexcept {
        return sizes_.data();
    }

    constexpr size_type const* address() const noexcept {
        return sizes_.data();
    }

    constexpr auto rank() const noexcept { 
        return rank_; 
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

    constexpr auto front() const { 
        return sizes_.front(); 
    }


    template<class Index>
    constexpr auto operator[](Index index) const { 
        return sizes_[normalize(index)]; 
    }

protected:
    template<class Index>
    constexpr auto normalize(Index index) const { 
        auto bound = rank();
        if (index < 0) index += bound;
        assert(index >= 0  && index < bound && "Index out of bound");
        return index;
    }
    
private:
    rank_type rank_{0};
    std::array<size_type, limit> sizes_{};
};

constexpr bool operator==(Strides const& first, Strides const& second) {
    if (first.rank() != second.rank()) return false;
    for (Strides::rank_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
}

inline std::ostream& operator<<(std::ostream& os, Strides const& strides) {
    os << "Strides(";
    for (Strides::rank_type dimension = 0; dimension < strides.rank(); ++dimension) {
        os << static_cast<unsigned int>(strides[dimension]);
        if (dimension + 1 < strides.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

#endif // STRIDES_HPP