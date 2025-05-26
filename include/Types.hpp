#ifndef TYPES_HPP
#define TYPES_HPP

#include <iostream>
#include <cstdint>
#include <stdexcept>    
#include <any>
#include <cassert>

enum type : uint8_t {
    float32,
    float64,
    integer8,
    integer16,
    integer32,
    TYPES
};

struct Trait {
    using size_type = size_t;
    using retrieve_function = std::any(*)(void*);
    using assign_function = void(*)(void*, const std::any&);
    using compare_function = bool(*)(void*, const std::any&);
    using print_function = std::ostream&(*)(std::ostream&, void*); 

    const char* name;
    size_type size;
    retrieve_function retrieve;
    assign_function assign;
    compare_function compare;
    print_function print; 
};


template<typename T>
inline void assign(void* address, const std::any& value) {
    *reinterpret_cast<T*>(address) = std::any_cast<T>(value);
}

template<typename T>
inline bool compare(void* address, const std::any& value) {
    return *reinterpret_cast<T*>(address) == std::any_cast<T>(value);
}

template<typename T>
inline std::any retrieve(void* address) {
    return *reinterpret_cast<T*>(address); 
}

template<typename T>
inline std::ostream& print(std::ostream& os, void* address) {
    return os << *reinterpret_cast<T*>(address);
}

template<>
inline std::ostream& print<int8_t>(std::ostream& os, void* address) {
    return os << +(*reinterpret_cast<int8_t*>(address));
}

static constexpr Trait traits[TYPES] = {
    [float32] = {
        .name = "float32",
        .size = sizeof(float),
        .retrieve = retrieve<float>,
        .assign = assign<float>,
        .compare = compare<float>,
        .print = print<float>
    },

    [float64] = {
        .name = "float64",
        .size = sizeof(double),
        .retrieve = retrieve<double>,
        .assign = assign<double>,
        .compare = compare<double>,
        .print = print<double>
    },

    [integer8] = {
        .name = "integer8",
        .size = sizeof(int8_t),
        .retrieve = retrieve<int8_t>,
        .assign = assign<int8_t>,
        .compare = compare<int8_t>,
        .print = print<int8_t>  
    },

    [integer16] = {
        .name = "integer16",
        .size = sizeof(int16_t),
        .retrieve = retrieve<int16_t>,
        .assign = assign<int16_t>,
        .compare = compare<int16_t>,
        .print = print<int16_t>
    },
    
    [integer32] = {
        .name = "integer32",
        .size = sizeof(int32_t),
        .retrieve = retrieve<int32_t>,
        .assign = assign<int32_t>,
        .compare = compare<int32_t>,
        .print = print<int32_t>
    }
};

template<typename T>
inline T cast(std::any& retrieved) {
    return std::any_cast<T>(retrieved);
}

inline std::ostream& operator<<(std::ostream& os, const type type) {
    assert(type < TYPES && "Invalid type");
    os << traits[type].name;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, uint8_t value) {
    return os << static_cast<unsigned int>(value);
}

inline constexpr size_t dsizeof(type type) {
    return traits[type].size;
}

constexpr type promote(type first, type second) {
    assert(first < TYPES && second < TYPES && "Invalid type");
    if (first != second) 
        throw std::runtime_error("Type promotion rules not implemented yet");
    return first;
} 

#endif // TYPES_HPP