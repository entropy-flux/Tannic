#include "cpu/cmps.hpp"
#include <stdexcept>
#include <cmath>

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
    size_t ne = 1;
    for (uint8_t dim = 0; dim < src0->rank; ++dim) {
        ne *= src0->shape.sizes[dim];
    }

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
        case float32: return launchAllcloseKernel<float>(src0, src1, rtol, atol);
        case float64: return launchAllcloseKernel<double>(src0, src1, rtol, atol);
        default: throw std::runtime_error("Unsupported dtype");
    }
}

} // namespace cpu