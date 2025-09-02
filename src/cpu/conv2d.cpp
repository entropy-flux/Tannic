#include <array> 
#include <cstdlib>
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
void conv2dKernel(
    const S* x_ptr, const Wt* w_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out, size_t K_h, size_t K_w,
    size_t H_out, size_t W_out,
    const size_t pad[2], const size_t stride[2],
    const strides_t& xs, const strides_t& ws, const strides_t& ys
) {
    auto offs4 = [](const strides_t& s, size_t n, size_t c, size_t h, size_t w) {
        return n * s.sizes[0] + c * s.sizes[1] + h * s.sizes[2] + w * s.sizes[3];
    };

    size_t pad_h = pad[0], pad_w = pad[1];
    size_t stride_h = stride[0], stride_w = stride[1];

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    D acc = D(0);

                    ptrdiff_t base_h = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_h);
                    ptrdiff_t base_w = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_w);

                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < K_h; ++kh) {
                            ptrdiff_t ih = base_h + static_cast<ptrdiff_t>(kh);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(H_in)) continue;

                            for (size_t kw = 0; kw < K_w; ++kw) {
                                ptrdiff_t iw = base_w + static_cast<ptrdiff_t>(kw);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(W_in)) continue;

                                size_t x_off = offs4(xs, n, ci, ih, iw);
                                size_t w_off = offs4(ws, co, ci, kh, kw);
                                acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
                            }
                        }
                    }

                    size_t y_off = offs4(ys, n, co, oh, ow);
                    y_ptr[y_off] = acc;
                }
            }
        }
    }
}

template<typename S, typename Wt, typename D, typename B>
void conv2dKernelBiased(
    const S* x_ptr, const Wt* w_ptr, const B* b_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out, size_t K_h, size_t K_w,
    size_t H_out, size_t W_out,
    const size_t pad[2], const size_t stride[2],
    const strides_t& xs, const strides_t& ws, const strides_t& ys
) {
    auto offs4 = [](const strides_t& s, size_t n, size_t c, size_t h, size_t w) {
        return n * s.sizes[0] + c * s.sizes[1] + h * s.sizes[2] + w * s.sizes[3];
    };

    size_t pad_h = pad[0], pad_w = pad[1];
    size_t stride_h = stride[0], stride_w = stride[1];

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    D acc = static_cast<D>(b_ptr[co]);  // <-- start with bias

                    ptrdiff_t base_h = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_h);
                    ptrdiff_t base_w = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_w);

                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < K_h; ++kh) {
                            ptrdiff_t ih = base_h + static_cast<ptrdiff_t>(kh);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(H_in)) continue;

                            for (size_t kw = 0; kw < K_w; ++kw) {
                                ptrdiff_t iw = base_w + static_cast<ptrdiff_t>(kw);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(W_in)) continue;

                                size_t x_off = offs4(xs, n, ci, ih, iw);
                                size_t w_off = offs4(ws, co, ci, kh, kw);
                                acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
                            }
                        }
                    }

                    size_t y_off = offs4(ys, n, co, oh, ow);
                    y_ptr[y_off] = acc;
                }
            }
        }
    }
}


template<typename S, typename Wt, typename D>
status launchConv2DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y,
    const size_t pad[2], const size_t stride[2]
) {
    if (x->rank != 4 || w->rank != 4 || y->rank != 4) {
        return ERROR;
    }

    size_t N     = x->shape.sizes[0];
    size_t C_in  = x->shape.sizes[1];
    size_t H_in  = x->shape.sizes[2];
    size_t W_in  = x->shape.sizes[3];

    size_t C_out = w->shape.sizes[0];
    size_t K_in  = w->shape.sizes[1];
    size_t K_h   = w->shape.sizes[2];
    size_t K_w   = w->shape.sizes[3];

    if (C_in != K_in) return ERROR;

    size_t H_out = (H_in + 2 * pad[0] - K_h) / stride[0] + 1;
    size_t W_out = (W_in + 2 * pad[1] - K_w) / stride[1] + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != H_out || y->shape.sizes[3] != W_out) {
            return ERROR;
    }

    conv2dKernel<S,Wt,D>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<D*>(y->address),
        N, C_in, H_in, W_in, C_out, K_h, K_w, H_out, W_out,
        pad, stride,
        x->strides, w->strides, y->strides
    );

    return SUCCESS;
}

template<typename S, typename Wt, typename D, typename B>
status launchConv2DKernelBiased(
    const tensor_t* x, const tensor_t* w, const tensor_t* b, tensor_t* y,
    const size_t pad[2], const size_t stride[2]
) {
    if (x->rank != 4 || w->rank != 4 || y->rank != 4 || b->rank != 1) {
        return ERROR;
    }

    size_t N     = x->shape.sizes[0];
    size_t C_in  = x->shape.sizes[1];
    size_t H_in  = x->shape.sizes[2];
    size_t W_in  = x->shape.sizes[3];

    size_t C_out = w->shape.sizes[0];
    size_t K_in  = w->shape.sizes[1];
    size_t K_h   = w->shape.sizes[2];
    size_t K_w   = w->shape.sizes[3];

    if (C_in != K_in || b->shape.sizes[0] != C_out) return ERROR;

    size_t H_out = (H_in + 2 * pad[0] - K_h) / stride[0] + 1;
    size_t W_out = (W_in + 2 * pad[1] - K_w) / stride[1] + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != H_out || y->shape.sizes[3] != W_out) {
            return ERROR;
    }

    conv2dKernelBiased<S,Wt,D,B>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<const B*>(b->address),
        static_cast<D*>(y->address),
        N, C_in, H_in, W_in, C_out, K_h, K_w, H_out, W_out,
        pad, stride,
        x->strides, w->strides, y->strides
    );

    return SUCCESS;
}


constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultConv2DKernel(
    const tensor_t*, const tensor_t*, tensor_t*, 
    const size_t[2], const size_t[2]
) {
    return UNSUPPORTED_DTYPE;
}

constexpr static status launchDefaultConv2DBiasedKernel(
    const tensor_t*, const tensor_t*, const tensor_t*, tensor_t*, 
    const size_t[2], const size_t[2]
) {
    return UNSUPPORTED_DTYPE;
}
 
using Conv2DKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, const size_t[2], const size_t[2]);
using Conv2DBiasedKernel = status(*)(const tensor_t*, const tensor_t*, const tensor_t*, tensor_t*, const size_t[2], const size_t[2]);

constexpr auto dispatchConv2D = []() {
    std::array<Conv2DKernel, index(TYPES, TYPES)> table;
    table.fill(launchDefaultConv2DKernel);
    table[index(int8, int8)] = launchConv2DKernel<int8_t, int8_t, int8_t>;
    table[index(int16, int16)] = launchConv2DKernel<int16_t, int16_t, int16_t>;
    table[index(int32, int32)] = launchConv2DKernel<int32_t, int32_t, int32_t>;
    table[index(int64, int64)] = launchConv2DKernel<int64_t, int64_t, int64_t>;
#if HAS_FLOAT16
    table[index(float16, float16)] = launchConv2DKernel<half, half, half>;
#endif
    table[index(float32, float32)] = launchConv2DKernel<float, float, float>;
    table[index(float64, float64)] = launchConv2DKernel<double, double, double>;
    return table;
}();
 
constexpr auto dispatchConv2DBiased = []() {
    std::array<Conv2DBiasedKernel, static_cast<int>(TYPES) * static_cast<int>(TYPES)> table;
    table.fill(launchDefaultConv2DBiasedKernel);
    table[index(int8, int8)]   = launchConv2DKernelBiased<int8_t, int8_t, int8_t, int8_t>;
    table[index(int16, int16)] = launchConv2DKernelBiased<int16_t, int16_t, int16_t, int16_t>;
    table[index(int32, int32)] = launchConv2DKernelBiased<int32_t, int32_t, int32_t, int32_t>;
    table[index(int64, int64)] = launchConv2DKernelBiased<int64_t, int64_t, int64_t, int64_t>;
#if HAS_FLOAT16
    table[index(float16, float16)] = launchConv2DKernelBiased<half, half, half, half>;
#endif
    table[index(float32, float32)] = launchConv2DKernelBiased<float, float, float, float>;
    table[index(float64, float64)] = launchConv2DKernelBiased<double, double, double, double>;
    return table;
}();


} namespace cpu {

status conv2d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, const size_t pad[2], const size_t stride[2]) {
    return dispatchConv2D[index(signal->dtype, kernel->dtype)](signal, kernel, dst, pad, stride);
}

status conv2d(const tensor_t* signal, const tensor_t* kernel, const tensor_t* bias, tensor_t* dst, const size_t pad[2], const size_t stride[2]) {
    return dispatchConv2DBiased[index(signal->dtype, kernel->dtype)](
        signal, kernel, bias, dst, pad, stride
    );
}

}