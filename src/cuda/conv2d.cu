#include <array>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include "cuda/conv.cuh"

namespace {

template<typename S, typename Wt, typename D>
__global__ void conv2dKernel(
    const S* x_ptr, const Wt* w_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out, size_t K_h, size_t K_w,
    size_t H_out, size_t W_out,
    size_t pad_h, size_t pad_w,
    size_t stride_h, size_t stride_w,
    strides_t xs, strides_t ws, strides_t ys
) {
    size_t n = blockIdx.z;
    size_t co = blockIdx.y;
    size_t oh = blockIdx.x / W_out;
    size_t ow = blockIdx.x % W_out;

    if (n >= N || co >= C_out || oh >= H_out || ow >= W_out) return;

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

                size_t x_off = n * xs.sizes[0] + ci * xs.sizes[1] + ih * xs.sizes[2] + iw * xs.sizes[3];
                size_t w_off = co * ws.sizes[0] + ci * ws.sizes[1] + kh * ws.sizes[2] + kw * ws.sizes[3];
                acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
            }
        }
    }

    size_t y_off = n * ys.sizes[0] + co * ys.sizes[1] + oh * ys.sizes[2] + ow * ys.sizes[3];
    y_ptr[y_off] = acc;
}

template<typename S, typename Wt, typename D>
status launchConv2DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y, stream_t stream, 
    const size_t pad[2], const size_t stride[2]
) {
    if (x->rank != 4 || w->rank != 4 || y->rank != 4) return ERROR;
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);

    size_t N = x->shape.sizes[0];
    size_t C_in = x->shape.sizes[1];
    size_t H_in = x->shape.sizes[2];
    size_t W_in = x->shape.sizes[3];

    size_t C_out = w->shape.sizes[0];
    size_t K_in = w->shape.sizes[1];
    size_t K_h = w->shape.sizes[2];
    size_t K_w = w->shape.sizes[3];

    if (C_in != K_in) return ERROR;

    size_t H_out = (H_in + 2 * pad[0] - K_h) / stride[0] + 1;
    size_t W_out = (W_in + 2 * pad[1] - K_w) / stride[1] + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != H_out || y->shape.sizes[3] != W_out) return ERROR;

    dim3 grid(W_out * H_out, C_out, N);
    dim3 block(1);

    conv2dKernel<S, Wt, D><<<grid, block, 0, cudaStream>>>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<D*>(y->address),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w, H_out, W_out,
        pad[0], pad[1],
        stride[0], stride[1],
        x->strides, w->strides, y->strides
    );

    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? SUCCESS : ERROR;
}

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultConv2DKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t, const size_t[2], const size_t[2]) {
    return UNSUPPORTED_DTYPE;
}

using Conv2DKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t, const size_t[2], const size_t[2]);

constexpr auto dispatchConv2DCUDA = []() {
    std::array<Conv2DKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultConv2DKernel);
    table[index(int8, int8)] = launchConv2DKernel<int8_t, int8_t, int8_t>; 
    table[index(float16, float16)] = launchConv2DKernel<__half, __half, __half>; 
    table[index(float32, float32)] = launchConv2DKernel<float, float, float>;
    table[index(float64, float64)] = launchConv2DKernel<double, double, double>;
    return table;
}();

} namespace cuda {  

status conv2d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, stream_t stream, const size_t pad[2], const size_t stride[2]) {
    return dispatchConv2DCUDA[index(signal->dtype, kernel->dtype)](signal, kernel, dst, stream, pad, stride);
}

}
