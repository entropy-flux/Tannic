#include <cstring>
#include "types.hpp"

namespace tannic {  

float16_t float32_to_float16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float)); 
    uint32_t sign     = (bits >> 16) & 0x8000;           
    int32_t exponent  = ((bits >> 23) & 0xFF) - 127 + 15; 
    uint32_t mantissa = bits & 0x007FFFFF;               

    uint16_t half;

    if (((bits >> 23) & 0xFF) == 0xFF) { 
        if (mantissa == 0) { 
            half = static_cast<uint16_t>(sign | 0x7C00);
        } else { 
            half = static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
            if ((half & 0x03FF) == 0) { 
                half |= 0x0001;
            }
        }
    } else if (exponent <= 0) { 
        if (exponent < -10) { 
            half = static_cast<uint16_t>(sign);
        } else { 
            mantissa |= 0x00800000; 
            int shift = 14 - exponent;
            uint32_t sub = mantissa >> shift; 
            if (mantissa & (1u << (shift - 1))) {
                sub += 1;
            }
            half = static_cast<uint16_t>(sign | sub);
        }
    } else if (exponent >= 31) { 
        half = static_cast<uint16_t>(sign | 0x7C00);
    } else { 
        half = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); 
        if (mantissa & 0x00001000) {
            half++;
        }
    }

    return float16_t{half};
} 

float float16_to_float32(float16_t value) {
    uint16_t half = value.bits; 
    uint32_t sign     = (half & 0x8000) << 16;       
    uint32_t exponent = (half & 0x7C00) >> 10;     
    uint32_t mantissa = (half & 0x03FF);            
    uint32_t bits; 
    if (exponent == 0) {
        if (mantissa == 0) { 
            bits = sign;
        } else { 
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;   
            exponent = exponent - 15 + 127;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) { 
        if (mantissa == 0) {
            bits = sign | 0x7F800000; 
        } else {
            bits = sign | 0x7F800000 | (mantissa << 13); 
        }
    } else { 
        exponent = exponent - 15 + 127;
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
} 

} 