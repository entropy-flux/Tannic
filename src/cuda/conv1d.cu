#include <array>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include "cuda/conv.cuh"

namespace {

template<typename S, typename Wt, typename D>
__global__ void conv1dKernel(
    const S* x_ptr, const Wt* w_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t L_in,
    size_t C_out, size_t K_l,
    size_t L_out,
    size_t pad_l,
    size_t stride_l,
    strides_t xs, strides_t ws, strides_t ys
) {
    size_t n = blockIdx.z;
    size_t co = blockIdx.y;
    size_t ol = blockIdx.x;

    if (n >= N || co >= C_out || ol >= L_out) return;

    D acc = D(0);
    ptrdiff_t base_l = static_cast<ptrdiff_t>(ol * stride_l) - static_cast<ptrdiff_t>(pad_l);

    for (size_t ci = 0; ci < C_in; ++ci) {
        for (size_t kl = 0; kl < K_l; ++kl) {
            ptrdiff_t il = base_l + static_cast<ptrdiff_t>(kl);
            if (il < 0 || il >= static_cast<ptrdiff_t>(L_in)) continue;

            size_t x_off = n * xs.sizes[0] + ci * xs.sizes[1] + il * xs.sizes[2];
            size_t w_off = co * ws.sizes[0] + ci * ws.sizes[1] + kl * ws.sizes[2];
            acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
        }
    }

    size_t y_off = n * ys.sizes[0] + co * ys.sizes[1] + ol * ys.sizes[2];
    y_ptr[y_off] = acc;
}

template<typename S, typename Wt, typename D>
status launchConv1DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y, stream_t stream, 
    const size_t pad, const size_t stride
) {
    if (x->rank != 3 || w->rank != 3 || y->rank != 3) return ERROR;
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);

    size_t N = x->shape.sizes[0];
    size_t C_in = x->shape.sizes[1];
    size_t L_in = x->shape.sizes[2];

    size_t C_out = w->shape.sizes[0];
    size_t K_in = w->shape.sizes[1];
    size_t K_l = w->shape.sizes[2];

    if (C_in != K_in) return ERROR;

    size_t L_out = (L_in + 2 * pad - K_l) / stride + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != L_out) return ERROR;

    dim3 grid(L_out, C_out, N);
    dim3 block(1);

    conv1dKernel<S, Wt, D><<<grid, block, 0, cudaStream>>>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<D*>(y->address),
        N, C_in, L_in,
        C_out, K_l,
        L_out,
        pad,
        stride,
        x->strides, w->strides, y->strides
    );

    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess ? SUCCESS : ERROR;
}

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultConv1DKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t, const size_t, const size_t) {
    return UNSUPPORTED_DTYPE;
}

using Conv1DKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t, const size_t, const size_t);

constexpr auto dispatchConv1DCUDA = []() {
    std::array<Conv1DKernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultConv1DKernel);
    table[index(int8, int8)] = launchConv1DKernel<int8_t, int8_t, int8_t>;
    table[index(float16, float16)] = launchConv1DKernel<__half, __half, __half>; 
    table[index(float32, float32)] = launchConv1DKernel<float, float, float>;
    table[index(float64, float64)] = launchConv1DKernel<double, double, double>;
    return table;
}();

} namespace cuda {  

status conv1d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, stream_t stream, const size_t pad, const size_t stride) {
    return dispatchConv1DCUDA[index(signal->dtype, kernel->dtype)](signal, kernel, dst, stream, pad, stride);
}

}