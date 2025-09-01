#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include <cuda_fp16.h> 
#include <thrust/complex.h>
#include "cuda/exc.cuh"
#include "cuda/ops.cuh"  

namespace {

template<typename S, typename D, class Op>
__global__ void singletonUnaryOpKernel(const S* __restrict__ src, D* __restrict__ dst, Op op) { 
    *dst = op(*src);
}  

template<typename S, typename D, class Op>
__global__ void contiguousUnaryOpKernel(const S* __restrict__ src, D* __restrict__ dst, size_t ne, Op op) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        dst[idx] = op(src[idx]);
    }
}
 
template<typename S, typename D, class Op>
__global__ void stridedUnaryOpKernel(
    const S* __restrict__ src_ptr, strides_t src_strides,    
    D* __restrict__ dst_ptr, shape_t resets,          
    uint8_t dst_rank, size_t ne, Op op
){
    int rank = static_cast<int>(dst_rank);
    const size_t gstride = size_t(blockDim.x) * gridDim.x;
    for (size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < ne; idx += gstride) {
        size_t offset = 0;
        size_t divisor = 1;

        for (int dim = rank - 1; dim >= 0; --dim) { 
            const size_t extent    = resets.sizes[dim] / src_strides.sizes[dim];
            const size_t coord     = (idx / divisor) % extent; 
            offset += coord * src_strides.sizes[dim];
            divisor *= extent;
        }

        dst_ptr[idx] = op(src_ptr[offset]);
    }
} 
 
template<typename S, typename D, class Op, class ... Args>
status launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, stream_t stream, Args... args)  { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    Op op(std::forward<Args>(args)...);

    size_t ne = dst->size; 
    size_t blockSize = 256;
    size_t gridSize = (ne + blockSize - 1) / blockSize;

    switch (src->layout) {
        case SINGLETON: {
            singletonUnaryOpKernel<S, D, Op><<<1, 1, 0, cudaStream>>>(
                (const S*)(src->address),
                (D*)(dst->address),
                op
            ); 
            return SUCCESS;
        }

        case CONTIGUOUS: {
            contiguousUnaryOpKernel<S, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
                (const S*)(src->address),
                (D*)(dst->address),
                ne,
                op
            );
            return SUCCESS;
        }

        case STRIDED: {  
            strides_t strides{0};
            shape_t resets{0};
            for (int dim = 0; dim < src->rank; ++dim) {
                resets.sizes[dim] = dst->shape.sizes[dim] * src->strides.sizes[dim];
                strides.sizes[dim] = src->strides.sizes[dim];
            } 
            
            stridedUnaryOpKernel<S, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
                (const S*)(src->address), strides,
                (D*)(dst->address), resets,
                src->rank, ne,
                op
            );
            return SUCCESS;
        }

        default:
            return ERROR;
    } 
}   

#include <cuda_fp16.h>
#include <type_traits>

struct Neg { 
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return -a;
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __hneg(a);  
    } 
};

struct Cpy { 
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const noexcept {
        return a;
    }
};

struct Log { 
    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(logf(__half2float(a)));
    }
     
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return log(a);
    }
};

struct Exp { 
    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(expf(__half2float(a)));
    }
     
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return exp(a);
    }
};

struct Sqrt { 
    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(sqrtf(__half2float(a)));
    }
     
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return sqrt(a);
    }
};

struct Rsqrt {
    float eps; 
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return 1.0 / sqrt(a + eps);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(rsqrtf(__half2float(a) + eps));
    }
     
    __device__ __forceinline__ float operator()(float a) const {
        return rsqrtf(a + eps);
    } 
};

struct Abs {
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return abs(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __habs(a);
    } 
};

struct Sin {
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return sin(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(sinf(__half2float(a)));
    } 
};

struct Cos {
    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return cos(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(cosf(__half2float(a)));
    } 
};

struct Tan {

    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return tan(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(tanf(__half2float(a)));
    } 
};

struct Sinh {

    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return sinh(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(sinhf(__half2float(a)));
    } 
};

struct Cosh {

    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return cosh(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(coshf(__half2float(a)));
    }
     
};

struct Tanh { 

    template<class A>
    __device__ __forceinline__ auto operator()(A a) const {
        return tanh(a);
    }

    __device__ __forceinline__ __half operator()(__half a) const {
        return __float2half(tanhf(__half2float(a)));
    } 
};

} namespace cuda {

status neg(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case int8:
            return launchUnaryOpKernel<int8_t, int8_t, Neg>(src, dst, stream);
        case int16:
            return launchUnaryOpKernel<int16_t, int16_t, Neg>(src, dst, stream);
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Neg>(src, dst, stream);
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Neg>(src, dst, stream);
        case float16:
            return launchUnaryOpKernel<__half, __half, Neg>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Neg>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Neg>(src, dst, stream);
        case complex64:
            return launchUnaryOpKernel<thrust::complex<float>, thrust::complex<float>, Neg>(src, dst, stream);
        case complex128:
            return launchUnaryOpKernel<thrust::complex<double>, thrust::complex<double>, Neg>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cpy(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case int8:
            return launchUnaryOpKernel<int8_t, int8_t, Cpy>(src, dst, stream);
        case int16:
            return launchUnaryOpKernel<int16_t, int16_t, Cpy>(src, dst, stream); 
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Cpy>(src, dst, stream); 
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Cpy>(src, dst, stream); 
        case float16:
            return launchUnaryOpKernel<__half, __half, Cpy>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Cpy>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Cpy>(src, dst, stream);
        case complex64:
            return launchUnaryOpKernel<thrust::complex<float>, thrust::complex<float>, Cpy>(src, dst, stream);
        case complex128:
            return launchUnaryOpKernel<thrust::complex<double>, thrust::complex<double>, Cpy>(src, dst, stream);
        default: 
            return UNSUPPORTED_DTYPE;
    }
}

status log(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Log>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Log>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Log>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status exp(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Exp>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Exp>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Exp>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sqrt(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Sqrt>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Sqrt>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Sqrt>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status rsqrt(const tensor_t* src, tensor_t* dst, stream_t stream, float eps) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Rsqrt>(src, dst, stream, eps);
        case float32:
            return launchUnaryOpKernel<float, float, Rsqrt>(src, dst, stream, eps);
        case float64:
            return launchUnaryOpKernel<double, double, Rsqrt>(src, dst, stream, eps);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status abs(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Abs>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Abs>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Abs>(src, dst, stream);
        case int32:
            return launchUnaryOpKernel<int32_t, int32_t, Abs>(src, dst, stream);
        case int64:
            return launchUnaryOpKernel<int64_t, int64_t, Abs>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sin(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Sin>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Sin>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Sin>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cos(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Cos>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Cos>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Cos>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tan(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Tan>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Tan>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Tan>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status sinh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Sinh>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Sinh>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Sinh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status cosh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Cosh>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Cosh>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Cosh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
}

status tanh(const tensor_t* src, tensor_t* dst, stream_t stream) {
    switch (src->dtype) {
        case float16:
            return launchUnaryOpKernel<__half, __half, Tanh>(src, dst, stream);
        case float32:
            return launchUnaryOpKernel<float, float, Tanh>(src, dst, stream);
        case float64:
            return launchUnaryOpKernel<double, double, Tanh>(src, dst, stream);
        default:
            return UNSUPPORTED_DTYPE;
    }
} 

} // namespace cuda