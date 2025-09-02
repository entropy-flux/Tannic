#include <array>
#include "cpu/conv.hpp"
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

template<typename S, typename Wt, typename D>
void conv1dKernel(
    const S* x_ptr, const Wt* w_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t L_in,
    size_t C_out, size_t K_l,
    size_t L_out,
    const size_t pad, const size_t stride,
    const strides_t& xs, const strides_t& ws, const strides_t& ys
) {
    auto offs3 = [](const strides_t& s, size_t n, size_t c, size_t l) {
        return n * s.sizes[0] + c * s.sizes[1] + l * s.sizes[2];
    };

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t ol = 0; ol < L_out; ++ol) {
                D acc = D(0);

                ptrdiff_t base_l = static_cast<ptrdiff_t>(ol * stride) - static_cast<ptrdiff_t>(pad);

                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t kl = 0; kl < K_l; ++kl) {
                        ptrdiff_t il = base_l + static_cast<ptrdiff_t>(kl);
                        if (il < 0 || il >= static_cast<ptrdiff_t>(L_in)) continue;

                        size_t x_off = offs3(xs, n, ci, il);
                        size_t w_off = offs3(ws, co, ci, kl);
                        acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
                    }
                }

                size_t y_off = offs3(ys, n, co, ol);
                y_ptr[y_off] = acc;
            }
        }
    }
}

template<typename S, typename Wt, typename D>
status launchConv1DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y,
    const size_t pad, const size_t stride
) {
    if (x->rank != 3 || w->rank != 3 || y->rank != 3) {
        return ERROR;
    }

    size_t N     = x->shape.sizes[0];
    size_t C_in  = x->shape.sizes[1];
    size_t L_in  = x->shape.sizes[2];

    size_t C_out = w->shape.sizes[0];
    size_t K_in  = w->shape.sizes[1];
    size_t K_l   = w->shape.sizes[2];

    if (C_in != K_in) return ERROR;

    size_t L_out = (L_in + 2 * pad - K_l) / stride + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != L_out) {
            return ERROR;
    }

    conv1dKernel<S,Wt,D>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<D*>(y->address),
        N, C_in, L_in, C_out, K_l, L_out,
        pad, stride,
        x->strides, w->strides, y->strides
    );

    return SUCCESS;
}

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultConv1DKernel(
    const tensor_t*, const tensor_t*, tensor_t*, 
    const size_t, const size_t
) {
    return UNSUPPORTED_DTYPE;
}

using Conv1DKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, const size_t, const size_t);

constexpr auto dispatchConv1D = []() {
    std::array<Conv1DKernel, index(TYPES, TYPES)> table;
    table.fill(launchDefaultConv1DKernel);
    table[index(int8, int8)] = launchConv1DKernel<int8_t, int8_t, int8_t>;
    table[index(int16, int16)] = launchConv1DKernel<int16_t, int16_t, int16_t>;
    table[index(int32, int32)] = launchConv1DKernel<int32_t, int32_t, int32_t>;
    table[index(int64, int64)] = launchConv1DKernel<int64_t, int64_t, int64_t>;
#if HAS_FLOAT16
    table[index(float16, float16)] = launchConv1DKernel<half, half, half>;
#endif
    table[index(float32, float32)] = launchConv1DKernel<float, float, float>;
    table[index(float64, float64)] = launchConv1DKernel<double, double, double>;
    return table;
}();

} namespace cpu {

status conv1d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, const size_t pad, const size_t stride) {
    return dispatchConv1D[index(signal->dtype, kernel->dtype)](signal, kernel, dst, pad, stride);
}

}