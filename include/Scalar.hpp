/*
#ifndef SCALAR_HPP
#define SCALAR_HPP

#include "Types.hpp"
#include "Resources.hpp"

#include <cstring>
#include <utility>

class Scalar {
public:
    template<typename T>
    Scalar(T value)
    :   dtype_(dtypeof<T>()) {
        address_ = ::operator new(sizeof(T));
        std::memcpy(address_, &value, sizeof(T));
    }

    template<typename U>
    Scalar(U value, type dtype)
    :   dtype_(dtype) {
        address_ = ::operator new(dsizeof(dtype));
        try {
            switch (dtype) {
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

    Scalar& operator=(const Scalar& other) {
        if (this != &other) {
            ::operator delete(address_);
            dtype_ = other.dtype_;
            address_ = ::operator new(sizeof(other.dtype_)); 
            std::memcpy(address_, other.address_, sizeof(other.dtype_));
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

private:
    void* address_;
    type dtype_;
};

#endif // SCALAR_HPP

*/