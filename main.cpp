#include <iostream>
#include <cstdint>
#include <utility>
#include <cstring>
#include <cassert>

enum type { 
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    complex64,   
    complex128,  
    TYPES
}; 

template<typename T> 
constexpr inline type dtypeof() {
    assert(sizeof(T) == 0 && "Only scalar types supported");
}

template<> constexpr inline type dtypeof<int8_t>()  { return int8; }
template<> constexpr inline type dtypeof<int16_t>() { return int16; }
template<> constexpr inline type dtypeof<int32_t>() { return int32; }
template<> constexpr inline type dtypeof<int64_t>() { return int64; }
template<> constexpr inline type dtypeof<float>()   { return float32; }
template<> constexpr inline type dtypeof<double>()  { return float64; } 


constexpr inline std::size_t dsizeof(type type) {
    switch (type) { 
        case int8:      return sizeof(int8_t);
        case int16:     return sizeof(int16_t);
        case int32:     return sizeof(int32_t);
        case int64:     return sizeof(int64_t);
        case float32:   return sizeof(float);
        case float64:   return sizeof(double);
        case complex64: return 2 * sizeof(float);     
        case complex128:return 2 * sizeof(double);  
        default:        return 0;
    }
}

class Scalar {
public: 

    Scalar(type dtype)
    :   dtype_(dtype) {}

    template<typename T>
    void initialize(T value) {
        address_ = ::operator new(dsizeof(dtype_));
        try {
            switch (dtype_) {
                case int8:    new(address_) int8_t(value);  break;
                case int16:   new(address_) int16_t(value); break;
                case int32:   new(address_) int32_t(value); break;
                case int64:   new(address_) int64_t(value); break;
                case float32: new(address_) float(value);   break;
                case float64: new(address_) double(value);  break;
                default:
                    throw std::invalid_argument("Unsupported type");
            }
        } catch (...) {
            ::operator delete(address_);
            throw;
        }
    }

    template<typename U>
    Scalar(type dtype, U value) : dtype_(dtype) { 
        initialize(value); }

    ~Scalar() {
        if (address_) { 
            ::operator delete(address_);
        }
    }

    Scalar(const Scalar& other)
    :   dtype_(other.dtype_) { 
        address_ = ::operator new(dsizeof(other.dtype_));   
        std::memcpy(address_, other.address_, dsizeof(other.dtype_));  
    }

    Scalar(Scalar&& other) noexcept
    :   address_(std::exchange(other.address_, nullptr)),
        dtype_(std::exchange(other.dtype_, {})) 
    {}



    Scalar& operator=(Scalar&& other) noexcept {
        if (this != &other) {
            ::operator delete(address_);   
            address_ = std::exchange(other.address_, nullptr);
            dtype_ = std::exchange(other.dtype_, {});
        }
        return *this;
    }
 
    
    Scalar& operator=(Scalar&& other) noexcept {
        if (this != &other) {
            ::operator delete(address_);
            address_ = std::exchange(other.address_, nullptr);
            dtype_ = std::exchange(other.dtype_, {});
        }
        return *this;
    }

    type dtype() const {
        return dtype_;
    }

private:
    void* address_ = nullptr;
    type dtype_;
};

int main() { 
    Scalar s1(int32); 
    Scalar s2(float32, 42);
    assert(s1.dtype() == int32);
    assert(s2.dtype() == float32);
    return 0;
}