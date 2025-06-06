#include "kernels/cpu/ops.h"

namespace cpu::negation {
    
template <typename T>
void neg(const T* in, T* out, size_t count) {
    for (size_t i = 0; i < count; ++i) out[i] = -in[i];
}

void neg_float32(const void* input, void* out, size_t count) {
    neg(static_cast<const float*>(input), static_cast<float*>(out), count);
}

void neg_float64(const void* input, void* out, size_t count) {
    neg(static_cast<const double*>(input), static_cast<double*>(out), count);
}

}  // namespace cpu::negation