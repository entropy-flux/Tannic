#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <array>
#include <cstddef>

#include "cuda/ops.cuh"

namespace {

template<typename S0, typename S1, typename D, class Op>
__global__ void contiguousScaleKernel(
    const S0* src0_ptr,
    S1 scalar_value,
    D* dst_ptr,
    size_t ne
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        Op op{};
        dst_ptr[idx] = static_cast<D>(op(src0_ptr[idx], scalar_value));
    }
}

template<typename S0, typename S1, typename D, class Op>
__global__ void stridedScaleOpKernel(
    const S0* src0_ptr, shape_t src0_shape, strides_t src0_strides,
    S1 scalar_value,
    D* dst_ptr,
    uint8_t rank,
    size_t ne
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ne) return;

    Op op{};
    size_t cnt[8] = {0};
    size_t tmp = idx;
    for (int dim = rank - 1; dim >= 0; --dim) {
        cnt[dim] = tmp % src0_shape.sizes[dim];
        tmp /= src0_shape.sizes[dim];
    }

    size_t offs0 = 0;
    for (int dim = 0; dim < rank; ++dim) {
        size_t coord = (src0_shape.sizes[dim] == 1) ? 0 : cnt[dim];
        offs0 += coord * src0_strides.sizes[dim];
    }

    dst_ptr[idx] = static_cast<D>(op(src0_ptr[offs0], scalar_value));
}

template<typename S0, typename S1, typename D, class Op>
status launchScaleOpKernel(const tensor_t* src0, const scalar_t* src1, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    const size_t ne = dst->size;
    S1 scalar = *reinterpret_cast<const S1*>(src1->address);

    const int threads = 256;
    const int blocks = (ne + threads - 1) / threads;

    if (src0->layout == CONTIGUOUS) {
        contiguousScaleKernel<S0, S1, D, Op><<<blocks, threads, 0, cudaStream>>>(
            reinterpret_cast<const S0*>(src0->address),
            scalar,
            reinterpret_cast<D*>(dst->address),
            ne
        );
    } else {
        shape_t src_shape;
        strides_t src_strides;
        for (int dim = 0; dim < src0->rank; ++dim) {
            src_shape.sizes[dim] = dst->shape.sizes[dim];
            src_strides.sizes[dim] = src0->strides.sizes[dim];
        }
        stridedScaleOpKernel<S0, S1, D, Op><<<blocks, threads, 0, cudaStream>>>(
            reinterpret_cast<const S0*>(src0->address),
            src_shape,
            src_strides,
            scalar,
            reinterpret_cast<D*>(dst->address),
            dst->rank,
            ne
        );
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SUCCESS : ERROR;
}

struct Mul {
    template<class A, class B>
    __device__ __host__ inline auto operator()(A a, B b) const noexcept(noexcept(a * b)) {
        return a * b;
    }
};

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultScaleOpKernel(const tensor_t* src0, const scalar_t* src1, tensor_t* dst, stream_t) {
    return UNSUPPORTED_DTYPE;
}

using Kernel = status(*)(const tensor_t*, const scalar_t*, tensor_t*, stream_t);

constexpr auto dispatchMul = []() {
    std::array<Kernel, index(TYPES, TYPES)> table;
    table.fill(launchDefaultScaleOpKernel);

    table[index(int8 , int8)]    = launchScaleOpKernel<uint8_t, uint8_t, uint8_t, Mul>;
    table[index(int16, int16)]   = launchScaleOpKernel<uint16_t, uint16_t, uint16_t, Mul>;
    table[index(int32, int32)]   = launchScaleOpKernel<uint32_t, uint32_t, uint32_t, Mul>;
    table[index(int64, int64)]   = launchScaleOpKernel<uint64_t, uint64_t, uint64_t, Mul>;
    table[index(float16, float16)]   = launchScaleOpKernel<__half, __half, __half, Mul>;
    table[index(bfloat16, bfloat16)] = launchScaleOpKernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, Mul>;
    table[index(float32, float32)] = launchScaleOpKernel<float, float, float, Mul>;
    table[index(float64, float64)] = launchScaleOpKernel<double, double, double, Mul>;
    return table;
}();

} namespace cuda {

status scale(const tensor_t* src0, const scalar_t* src1, tensor_t* dst, stream_t stream) {
    return dispatchMul[index(src0->dtype, src1->dtype)](src0, src1, dst, stream);
}

}
