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
    static constexpr uint8_t limit = 6;  

 
    template<typename... Steps>
    constexpr explicit Strides(Steps... steps)
    : steps_{static_cast<step_type>(steps)...}
    , rank_(sizeof...(steps)) {
        if (sizeof...(steps) > limit) {
            throw "Rank limit exceeded";
        }
    }
  
    template<std::input_iterator Iterator>
    constexpr Strides(Iterator begin, Iterator end) { 
        step_type dimension = 0;
        for (auto iterator = begin; iterator != end; ++iterator) {
            if (dimension >= limit) {
                throw "Strides rank limit exceeded";
            }
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

    constexpr step_type rank() const noexcept { return rank_; }
    constexpr step_type operator[](rank_type dimension) const noexcept { return steps_[dimension]; }
    constexpr step_type& operator[](rank_type dimension) noexcept { return steps_[dimension]; }

    constexpr auto begin() { return steps_.begin(); }
    constexpr auto end() { return steps_.begin() + rank_; }

    constexpr auto begin() const { return steps_.begin(); }
    constexpr auto end() const { return steps_.begin() + rank_; }

    constexpr auto cbegin() const { return steps_.cbegin(); }
    constexpr auto cend() const { return steps_.cbegin() + rank_; }

    constexpr auto front() const { return steps_.front(); }

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