#ifndef SHAPE_HPP
#define SHAPE_HPP
 
#include <type_traits> 
#include <array>
#include <cstdint>

template<typename T>
concept Iterable = requires(T type) {
    std::begin(type);
    std::end(type);
};

class Shape {
public:
    using rank_type = uint8_t;
    using size_type = std::size_t;
    static constexpr size_type limit = 6;  

    template<typename... Sizes>
    constexpr explicit Shape(Sizes... sizes) 
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {      
        if (sizeof...(sizes) > limit) {
            throw "Rank limit exceeded";
        }
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
    }

    constexpr size_type rank() const noexcept { return rank_; }
    constexpr size_type operator[](rank_type dimension) const noexcept { return sizes_[dimension]; }
    constexpr size_type& operator[](rank_type dimension) noexcept { return sizes_[dimension]; }


    constexpr auto begin() { return sizes_.begin(); }
    constexpr auto end() { return sizes_.begin() + rank_; }

    constexpr auto begin() const { return sizes_.begin(); }
    constexpr auto end() const { return sizes_.begin() + rank_; }

    constexpr auto cbegin() const { return sizes_.cbegin(); }
    constexpr auto cend() const { return sizes_.cbegin() + rank_; }
    
private:
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

#endif // SHAPE_HPP