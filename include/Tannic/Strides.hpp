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
// This file is part of Tannic, a A C++ tensor library.  .

#ifndef STRIDES_HPP
#define STRIDES_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <ostream>

#include "Shape.hpp"

class Strides {
public:
    using step_type = std::size_t;
    using rank_type = Shape::rank_type;
    using index_type = Shape::index_type;
    static constexpr uint8_t limit = 8;  

    constexpr Strides() noexcept = default;
 
    template<typename... Steps>
    constexpr explicit Strides(Steps... steps)
    :   steps_{static_cast<step_type>(steps)...}
    ,   rank_(sizeof...(steps)) {
        assert(sizeof...(steps) < limit && "Strides rank limit exceeded");
    }
  
    template<std::input_iterator Iterator>
    constexpr Strides(Iterator begin, Iterator end) { 
        step_type dimension = 0;
        for (auto iterator = begin; iterator != end; ++iterator) {
            assert(dimension < limit && "Strides rank limit exceeded");
            steps_[dimension++] = static_cast<step_type>(*iterator);
        }
        rank_ = dimension;
    }
 

    constexpr explicit Strides(const Shape& shape) { 
        rank_ = shape.rank();
        if (rank_ == 0) return;
        
        steps_[rank_ - 1] = 1;
        for (int step = rank_ - 2; step >= 0; --step) {
            steps_[step] = steps_[step + 1] * shape[step + 1];
        }
    }
    

    constexpr rank_type normalize(index_type index, rank_type extra = 0) const { 
        rank_type bound = rank() + extra;
        if (index < 0) index += bound;
        assert(index >= 0  && index < bound && "Index out of bound");
        return static_cast<rank_type>(index);
    }

    constexpr step_type rank() const noexcept { return rank_; }
    constexpr step_type operator[](index_type dimension) const noexcept { return steps_[normalize(dimension)]; }
    constexpr step_type& operator[](index_type dimension) noexcept { return steps_[normalize(dimension)]; }

    constexpr auto begin() { return steps_.begin(); }
    constexpr auto end() { return steps_.begin() + rank_; }

    constexpr auto begin() const { return steps_.begin(); }
    constexpr auto end() const { return steps_.begin() + rank_; }

    constexpr auto cbegin() const { return steps_.cbegin(); }
    constexpr auto cend() const { return steps_.cbegin() + rank_; }

    constexpr auto front() const { return steps_.front(); }

    
    constexpr Strides transpose(index_type first, index_type second) const { 
        Strides result = *this;
        std::swap(result.steps_[normalize(first)], result.steps_[normalize(second)]);
        return result;
    }
    
private:
    std::array<step_type, limit> steps_{};
    rank_type rank_{0};
};

constexpr bool operator==(const Strides& first, const Strides& second) {
    if (first.rank() != second.rank()) return false;
    for (Strides::rank_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
}

inline std::ostream& operator<<(std::ostream& os, const Strides& strides) {
    os << "Strides(";
    for (Strides::rank_type dimension = 0; dimension < strides.rank(); ++dimension) {
        os << strides[dimension];
        if (dimension + 1 < strides.rank()) {
            os << ", ";
        }
    }
    os << ")";
    return os;
}

#endif // STRIDES_HPP