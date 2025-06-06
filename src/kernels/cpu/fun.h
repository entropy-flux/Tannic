#pragma once
#include <cstddef>
#include <array>
#include "Types.hpp"

namespace cpu {

// LOG (with base param)
namespace log {
    void log_float32(const void*, void*, size_t);
    void log_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = log_float32;
        kernels[float64] = log_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// EXP
namespace exp {
    void exp_float32(const void*, void*, size_t);
    void exp_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = exp_float32;
        kernels[float64] = exp_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// SQRT
namespace sqrt {
    void sqrt_float32(const void*, void*, size_t);
    void sqrt_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = sqrt_float32;
        kernels[float64] = sqrt_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// ABS
namespace abs {
    void abs_float32(const void*, void*, size_t);
    void abs_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = abs_float32;
        kernels[float64] = abs_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// SIN
namespace sin {
    void sin_float32(const void*, void*, size_t);
    void sin_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = sin_float32;
        kernels[float64] = sin_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// SINH
namespace sinh {
    void sinh_float32(const void*, void*, size_t);
    void sinh_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = sinh_float32;
        kernels[float64] = sinh_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// COS
namespace cos {
    void cos_float32(const void*, void*, size_t);
    void cos_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = cos_float32;
        kernels[float64] = cos_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// COSH
namespace cosh {
    void cosh_float32(const void*, void*, size_t);
    void cosh_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = cosh_float32;
        kernels[float64] = cosh_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// TAN
namespace tan {
    void tan_float32(const void*, void*, size_t);
    void tan_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = tan_float32;
        kernels[float64] = tan_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

// TANH
namespace tanh {
    void tanh_float32(const void*, void*, size_t);
    void tanh_float64(const void*, void*, size_t);

    using Kernel = void(*)(const void*, void*, size_t);
    using Kernels = std::array<Kernel, TYPES>;

    constexpr Kernels make_kernels() {
        Kernels kernels{};
        kernels[float32] = tanh_float32;
        kernels[float64] = tanh_float64;
        return kernels;
    }

    inline constexpr auto kernels = make_kernels();
}

}  // namespace cpu 