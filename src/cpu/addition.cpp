#include "cpu/kernels.h"

namespace cpu::addition {
    
template <typename L, typename R, typename O>
void add(const L* l, const R* r, O* o, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        o[i] = l[i] + r[i];  
    }
}

void add_float32_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    add(
        static_cast<const float*>(lhs),
        static_cast<const float*>(rhs),
        static_cast<float*>(out),
        count
    );
}

void add_float64_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    add(
        static_cast<const double*>(lhs),
        static_cast<const double*>(rhs),
        static_cast<double*>(out),
        count
    );
}

void add_float32_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    add(
        static_cast<const float*>(lhs),
        static_cast<const double*>(rhs),
        static_cast<double*>(out),
        count
    );
}

void add_float64_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    add(
        static_cast<const double*>(lhs),
        static_cast<const float*>(rhs),
        static_cast<double*>(out),
        count
    );
}

} // namespace cpu::addition