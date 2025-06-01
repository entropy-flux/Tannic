#include "cpu/kernels.h"

namespace cpu::subtraction {

template <typename L, typename R, typename O>
void sub(const L* l, const R* r, O* o, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        o[i] = l[i] - r[i];
    }
}

void sub_float32_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    sub(
        static_cast<const float*>(lhs),
        static_cast<const float*>(rhs),
        static_cast<float*>(out),
        count
    );
}

void sub_float64_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    sub(
        static_cast<const double*>(lhs),
        static_cast<const double*>(rhs),
        static_cast<double*>(out),
        count
    );
}

void sub_float32_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    sub(
        static_cast<const float*>(lhs),
        static_cast<const double*>(rhs),
        static_cast<double*>(out),
        count
    );
}

void sub_float64_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    sub(
        static_cast<const double*>(lhs),
        static_cast<const float*>(rhs),
        static_cast<double*>(out),
        count
    );
}

} // namespace cpu::subtraction
