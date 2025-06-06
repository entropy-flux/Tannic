#include <cmath>
#include "kernels/cpu/fun.h"

namespace cpu {

namespace log {
    void log_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::log(input[i]);
    }

    void log_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::log(input[i]);
    }
}

namespace exp {
    void exp_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::exp(input[i]);
    }

    void exp_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::exp(input[i]);
    }
}

namespace sqrt {
    void sqrt_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sqrt(input[i]);
    }

    void sqrt_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sqrt(input[i]);
    }
}

namespace abs {
    void abs_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::fabs(input[i]);
    }

    void abs_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::fabs(input[i]);
    }
}

namespace sin {
    void sin_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sin(input[i]);
    }

    void sin_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sin(input[i]);
    }
}

namespace sinh {
    void sinh_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sinh(input[i]);
    }

    void sinh_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::sinh(input[i]);
    }
}

namespace cos {
    void cos_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::cos(input[i]);
    }

    void cos_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::cos(input[i]);
    }
}

namespace cosh {
    void cosh_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::cosh(input[i]);
    }

    void cosh_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::cosh(input[i]);
    }
}

namespace tan {
    void tan_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::tan(input[i]);
    }

    void tan_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::tan(input[i]);
    }
}

namespace tanh {
    void tanh_float32(const void* in, void* out, size_t size) {
        const float* input = static_cast<const float*>(in);
        float* output = static_cast<float*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::tanh(input[i]);
    }

    void tanh_float64(const void* in, void* out, size_t size) {
        const double* input = static_cast<const double*>(in);
        double* output = static_cast<double*>(out);
        for (size_t i = 0; i < size; ++i)
            output[i] = std::tanh(input[i]);
    }
}

} // namespace cpu