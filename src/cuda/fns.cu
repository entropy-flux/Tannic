#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/fns.cuh"  

namespace {
    
template<typename S, typename D, class Fn>
__global__ void scalarFnKernel(const S* src, D* dst, Fn fn) { 
    *dst = fn(*src);
}  

template<typename S, typename D, class Fn>
__global__ void batchedFnKernel(
    const S* src, shape_t src_shape, strides_t src_strides,
    D* dst, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne, Fn fn
) { 
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) { 
        size_t offs = 0;
        size_t remaining = idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % dst_shape.sizes[dim];
            remaining /= dst_shape.sizes[dim];
 
            size_t src_idx = (src_shape.sizes[dim] == 1) ? 0 : dim_idx;
            offs += src_idx * src_strides.sizes[dim];
        }

        dst[idx] = fn(src[offs]);
    }
}   

template<typename S, typename D, class Fn, class ... Args>
status launchFnKernel(const tensor_t* src, tensor_t* dst, stream_t stream, Args... args)  { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    Fn fn(std::forward<Args>(args)...);
    if (src->rank == 0) {
        scalarFnKernel<S, D, Fn><<<1, 1, 0, cudaStream>>>(
            (const S*)(src->address),
            (D*)(dst->address), fn
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedFnKernel<S, D, Fn><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne, fn
        ); 
    }  
    return SUCCESS;
} 
 
struct Log { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(log(a))) {
        return log(a);
    }
};
 
struct Exp { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(exp(a))) {
        return exp(a);
    }
};
  
struct Sqrt { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sqrt(a))) {
        return sqrt(a);
    }
};

struct Rsqrt {
    float eps; 

    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept {
        if constexpr (std::is_same_v<std::decay_t<A>, float>) {
            return rsqrtf(a + eps);
        } else {
            return 1.0 / sqrt(a + eps);
        }
    }
}; 

struct Abs { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(abs(a))) {
        return abs(a);
    }
};
 
struct Sin { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sin(a))) {
        return sin(a);
    }
};
 
struct Cos { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(cos(a))) {
        return cos(a);
    }
};

struct Tan { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(tan(a))) {
        return tan(a);
    }
}; 

struct Sinh { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sinh(a))) {
        return sinh(a);
    }
};
 
struct Cosh{ 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(cosh(a))) {
        return cosh(a);
    }
};
 
struct Tanh { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(tanh(a))) {
        return tanh(a);
    }
};    

} namespace cuda {

status log(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Log>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Log>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Exp>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Exp>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sqrt>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Sqrt>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}


status rsqrt(const tensor_t* src, tensor_t* dst, stream_t stream, float eps) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Rsqrt>(src, dst, stream, eps);
        case float64:
            return launchFnKernel<double, double, Rsqrt>(src, dst, stream, eps);
        default:
            return UNSUPPORTED_DTYPE;
    }
}


status abs(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Abs>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Abs>(src, dst, stream);
        case int32:
            return launchFnKernel<int32_t, int32_t, Abs>(src, dst, stream);
        case int64:
            return launchFnKernel<int64_t, int64_t, Abs>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sin>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Sin>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Cos>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Cos>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Tan>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Tan>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Sinh>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Sinh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Cosh>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Cosh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float32:
            return launchFnKernel<float, float, Tanh>(src, dst, stream);
        case float64:
            return launchFnKernel<double, double, Tanh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

} // namespace cuda
