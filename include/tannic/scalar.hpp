#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <iostream> 
#include <variant>
#include <complex>
#include <cstdint>
#include <iostream>  
#include <tannic.hpp>  
 
namespace tannic {

class Scalar {
public:
    using Value = std::variant< 
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        float16_t,
        bfloat16_t,
        float,                 
        double,             
        std::complex<float>, 
        std::complex<double> 
    >;

    template <typename T>
    constexpr Scalar(T&& value) 
    :   dtype_(dtypeof<T>())   
    ,   value_(std::forward<T>(value))  
    {}

    template <typename T>
    constexpr Scalar(T&& value, type dtype) 
    :   dtype_(dtype) {
        switch (dtype) { 
            case int8:       value_ = static_cast<int8_t>(value);  break;
            case int16:      value_ = static_cast<int16_t>(value); break;
            case int32:      value_ = static_cast<int32_t>(value); break;
            case int64:      value_ = static_cast<int64_t>(value); break;
            case float16:    value_ = static_cast<float16_t>(value);  break;
            case bfloat16:   value_ = static_cast<bfloat16_t>(value); break;
            case float32:    value_ = static_cast<float>(value);   break;
            case float64:    value_ = static_cast<double>(value);  break;
            case complex64:  value_ = std::complex<float>(value);  break;
            case complex128: value_ = std::complex<double>(value); break;
            default:         throw Exception("Unsupported Scalar dtype");
        }
    }

    template <typename T>
    constexpr T get() const { return std::get<T>(value_); }  

    constexpr type dtype() const {
        return dtype_;
    }
    
private:
    type dtype_;
    Value value_;
};

} // namespace tannic
 
#endif // SCALAR_HPP