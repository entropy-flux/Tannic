#include <iostream>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>  // CUDA half support
 
struct float16_t {
    uint16_t data;
    float16_t() = default;
    float16_t(float f) { data = float_to_half(f); }
    operator float() const { return half_to_float(data); } 
private:
    static uint16_t float_to_half(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(f));
        uint32_t sign = (bits >> 16) & 0x8000;
        uint32_t mant = bits & 0x007FFFFF;
        int32_t exp  = ((bits >> 23) & 0xFF) - 127 + 15;
        if (exp <= 0) return static_cast<uint16_t>(sign);
        else if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
        return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
    }
    static float half_to_float(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x03FF;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) bits = sign;
            else {
                exp = 127 - 14;
                while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
                mant &= 0x03FF;
                bits = sign | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000 | (mant << 13); // Inf/NaN
        } else {
            exp = exp - 15 + 127;
            bits = sign | (exp << 23) | (mant << 13);
        }
        float f; std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// --- bridge helpers ---
inline __half to_cuda_half(const float16_t& h) {
    __half out;
    std::memcpy(&out, &h.data, sizeof(uint16_t)); // copy raw 16 bits
    return out;
}

inline float16_t from_cuda_half(const __half& h) {
    float16_t out;
    std::memcpy(&out.data, &h, sizeof(uint16_t));
    return out;
}


#include <cstdint>
#include <iostream>

struct int4_t {
    int8_t value; // store as 8-bit, but only use low 4 bits

    int4_t() : value(0) {}
    int4_t(int v) { set(v); }

    void set(int v) {
        if (v > 7) v = 7;
        if (v < -8) v = -8;
        value = static_cast<int8_t>(v & 0x0F); // mask to 4 bits
    }

    operator int() const {
        // sign extend 4-bit value
        int8_t v = value & 0x0F;
        if (v & 0x08) v |= 0xF0; // extend negative sign
        return static_cast<int>(v);
    }
};

inline uint8_t pack_int4(int4_t a, int4_t b) {
    uint8_t lo = static_cast<uint8_t>(int(a) & 0x0F);
    uint8_t hi = static_cast<uint8_t>(int(b) & 0x0F) << 4;
    return lo | hi;
}

inline std::pair<int4_t, int4_t> unpack_int4(uint8_t byte) {
    int4_t a(byte & 0x0F);          // lower 4 bits
    int4_t b((byte >> 4) & 0x0F);   // upper 4 bits
    return {a, b};
}


int main() {
    float f = 3.14159f;
    float16_t h_cpu(f);          // float -> CPU half
    __half h_gpu = to_cuda_half(h_cpu);   // CPU half -> CUDA half
    float16_t back = from_cuda_half(h_gpu); // CUDA half -> CPU half

    std::cout << "original: " << f
              << "  -> cpu_half bits: 0x" << std::hex << h_cpu.data
              << "  -> roundtrip: " << std::dec << float(back)
              << "\n";


    int4_t a(-3), b(7);
    uint8_t packed = pack_int4(a, b);

    auto [ua, ub] = unpack_int4(packed);

    std::cout << "a=" << int(a) << " b=" << int(b) << "\n";
    std::cout << "packed=0x" << std::hex << int(packed) << std::dec << "\n";
    std::cout << "unpacked: ua=" << int(ua) << " ub=" << int(ub) << "\n";
}
