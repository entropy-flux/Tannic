#include "kernels/cpu/ops.h"

namespace cpu::multiplication {
template <typename L, typename R, typename O>
void mul(const L* l, const R* r, O* o, size_t size) {
    for (size_t i = 0; i < size; ++i) o[i] = l[i] * r[i];
}

void mul_float32_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    mul(static_cast<const float*>(lhs), static_cast<const float*>(rhs), static_cast<float*>(out), count);
}

void mul_float64_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    mul(static_cast<const double*>(lhs), static_cast<const double*>(rhs), static_cast<double*>(out), count);
}

void mul_float32_float64(const void* lhs, const void* rhs, void* out, size_t count) {
    mul(static_cast<const float*>(lhs), static_cast<const double*>(rhs), static_cast<double*>(out), count);
}

void mul_float64_float32(const void* lhs, const void* rhs, void* out, size_t count) {
    mul(static_cast<const double*>(lhs), static_cast<const float*>(rhs), static_cast<double*>(out), count);
}

}  // namespace cpu::multiplication
