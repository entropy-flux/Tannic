#include <cstring>
#include "types.hpp"

namespace tannic {  
    
float16_t::float16_t(float value) {
    uint32_t fbits;
    std::memcpy(&fbits, &value, sizeof(float));

    uint32_t sign     = (fbits >> 16) & 0x8000;
    int32_t exponent  = ((fbits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = fbits & 0x007FFFFF;
    uint16_t half;

    if (((fbits >> 23) & 0xFF) == 0xFF) {  
        if (mantissa == 0) {
            half = static_cast<uint16_t>(sign | 0x7C00);  
        } else {
            half = static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
            if ((half & 0x03FF) == 0) half |= 0x0001; 
        }
    } else if (exponent <= 0) { 
        if (exponent < -10) {
            half = static_cast<uint16_t>(sign); 
        } else {
            mantissa |= 0x00800000;
            int shift = 14 - exponent;
            uint32_t sub = mantissa >> shift;
            if (mantissa & (1u << (shift - 1))) sub++; 
            half = static_cast<uint16_t>(sign | sub);
        }
    } else if (exponent >= 31) { 
        half = static_cast<uint16_t>(sign | 0x7C00);
    } else { // Normal
        half = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
        if (mantissa & 0x00001000) half++; 
    }

    bits = half; 
}

float16_t::operator float() const {
    uint16_t half = bits;
    uint32_t sign     = (half & 0x8000) << 16;
    uint32_t exponent = (half & 0x7C00) >> 10;
    uint32_t mantissa = (half & 0x03FF);
    uint32_t fbits;

    if (exponent == 0) {
        if (mantissa == 0) {
            fbits = sign; 
        } else {
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            exponent = exponent - 15 + 127;
            fbits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        fbits = sign | 0x7F800000 | (mantissa << 13); // Inf/NaN
    } else {
        exponent = exponent - 15 + 127;
        fbits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &fbits, sizeof(float));
    return result;
}

} // namespace tannic
 