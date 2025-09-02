#include "cpu/cmps.hpp"
#include <stdexcept>
#include <cmath>
#ifndef HAS_FLOAT16
    #if defined(__STDCPP_FLOAT16_T__) && __STDCPP_FLOAT16_T__
        #include <stdfloat>
        using half = std::float16_t;
        #define HAS_FLOAT16 1
    #else 
        #define HAS_FLOAT16 0 
        struct half_placeholder { float value; };
        using half = half_placeholder;
    #endif
#endif

namespace {

template<typename T>
void allcloseKernel(
    const T* src0_ptr, const strides_t& src0_strides,
    const T* src1_ptr, const strides_t& src1_strides,
    const shape_t& shape, uint8_t rank, size_t ne,
    double rtol, double atol,
    bool* result
) {
    size_t indices[8] = {0};

    for (size_t linear_idx = 0; linear_idx < ne; ++linear_idx) {
        size_t off0 = 0, off1 = 0;
        for (size_t d = 0; d < rank; ++d) {
            off0 += indices[d] * src0_strides.sizes[d];
            off1 += indices[d] * src1_strides.sizes[d];
        }

        double a = static_cast<double>(src0_ptr[off0]);
        double b = static_cast<double>(src1_ptr[off1]);
        if (std::fabs(a - b) > (atol + rtol * std::fabs(b))) {
            *result = false;
            return;
        }

        // increment indices
        for (int d = rank - 1; d >= 0; --d) {
            if (++indices[d] < shape.sizes[d]) break;
            indices[d] = 0;
        }
    }
}

template<typename S>
bool launchAllcloseKernel(const tensor_t* src0, const tensor_t* src1,
                          double rtol, double atol) {
    size_t ne = src0->size;

    bool result = true;

    if (src0->rank == 0) {
        double a = static_cast<double>(*(const S*)(src0->address));
        double b = static_cast<double>(*(const S*)(src1->address));
        result = (std::fabs(a - b) <= (atol + rtol * std::fabs(b)));
    } else {
        allcloseKernel<S>(
            (const S*)(src0->address), src0->strides,
            (const S*)(src1->address), src1->strides,
            src0->shape, src0->rank, ne, rtol, atol,
            &result
        );
    }
    return result;
}

} namespace cpu {

bool allclose(const tensor_t* src0, const tensor_t* src1, double rtol, double atol) {
    switch (src0->dtype) { 
        case float16: return launchAllcloseKernel<half>(src0, src1, rtol, atol);
        case float32: return launchAllcloseKernel<float>(src0, src1, rtol, atol);
        case float64: return launchAllcloseKernel<double>(src0, src1, rtol, atol);
        default: throw std::runtime_error("Unsupported dtype");
    }
}

} // namespace cpu