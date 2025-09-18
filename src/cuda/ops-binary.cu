#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <thrust/complex.h>
#include "cuda/exc.cuh"
#include "cuda/ops.cuh" 

namespace { 

template<typename S0, typename S1, typename D, class Op>
__global__ void singletonBinaryOpKernel(const S0* src0, const S1* src1, D* dst) {
    Op op;
    *dst = op(*src0, *src1);
}  

template<typename S0, typename S1, typename D, class Op>
__global__ void flatBinaryOpKernel(const S0* src0, const S1* src1, D* dst, size_t ne, Op op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < ne; idx += stride) {
        dst[idx] = op(src0[idx], src1[idx]);
    }
} 

template<typename S0, typename S1, typename D, class Op>
__global__ void broadcastBinaryOpKernel(
    const S0* src0, strides_t src0_strides,
    const S1* src1, strides_t src1_strides,
    D* dst, strides_t dst_strides,
    shape_t shape, uint8_t rank,
    size_t ne, Op op)
{
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = blockDim.x * gridDim.x;
 
    for (; linear_idx < ne; linear_idx += grid_stride) {
        size_t offset0 = 0, offset1 = 0, offset_dst = 0;
        size_t remaining = linear_idx;   

        #pragma unroll
        for (int dim = rank - 1; dim >= 0; --dim) {
            const size_t coord = remaining % shape.sizes[dim];
            remaining /= shape.sizes[dim];

            offset0    += coord * (size_t)src0_strides.sizes[dim];
            offset1    += coord * (size_t)src1_strides.sizes[dim];
            offset_dst += coord * (size_t)dst_strides.sizes[dim];
        }

        dst[offset_dst] = op(src0[offset0], src1[offset1]);
    }
}



template<typename S0, typename S1, typename D, class Op>
status launchBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    const size_t ne = dst->size;
    std::cout << "Launching binary op kernel" << std::endl;
    if (dst->rank == 0) {
        singletonBinaryOpKernel<S0, S1, D, Op><<<1, 1, 0, cudaStream>>>(
            (const S0*)src0->address,
            (const S1*)src1->address,
            (D*)dst->address
        );
        return SUCCESS;
    }

    else {
        strides_t src0_strides{};
        strides_t src1_strides{};

        bool flat = true;

        int64_t expect_src0 = 1;
        int64_t expect_src1 = 1;

        const int off0 = dst->rank - src0->rank;
        const int off1 = dst->rank - src1->rank;

        for (int dim = dst->rank - 1; dim >= 0; --dim) {
            const size_t sz = dst->shape.sizes[dim]; 
            const size_t sz0 = (dim >= off0) ? src0->shape.sizes[dim - off0] : 1;
            const size_t sz1 = (dim >= off1) ? src1->shape.sizes[dim - off1] : 1;
 
            src0_strides.sizes[dim] = (sz0 == 1) ? 0 : src0->strides.sizes[dim - off0];
            src1_strides.sizes[dim] = (sz1 == 1) ? 0 : src1->strides.sizes[dim - off1];
 
            if (!(sz0 == sz && ((dim >= off0) ? src0->strides.sizes[dim - off0] == expect_src0 : expect_src0 == 1)))
                flat = false;
 
            if (!(sz1 == sz && ((dim >= off1) ? src1->strides.sizes[dim - off1] == expect_src1 : expect_src1 == 1)))
                flat = false;
 
            expect_src0 *= (sz0 == 1 ? 1 : sz0);
            expect_src1 *= (sz1 == 1 ? 1 : sz1);
        }

        const size_t blockSize = 256;
        const size_t gridSize  = (ne + blockSize - 1) / blockSize;

        if (flat) {
            flatBinaryOpKernel<S0, S1, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
                (const S0*)src0->address,
                (const S1*)src1->address,
                (D*)dst->address,
                ne,
                Op{}
            );
        } 
        
        else {
            broadcastBinaryOpKernel<S0, S1, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
                (const S0*)src0->address, src0_strides,
                (const S1*)src1->address, src1_strides,
                (D*)dst->address, dst->strides,
                dst->shape, dst->rank,
                ne,
                Op{}
            );
        }

        return SUCCESS;
    }
}

constexpr static status launchDefaultBinaryOpKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {
    return UNSUPPORTED_DTYPE;
};    


struct Add {

    template<class A, class B>
    __device__ __forceinline__ auto operator()(A a, B b) const {
        return a + b;
    }

    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __hadd(a, b);
    }
    
    __device__ __forceinline__ float operator()(__half a, float b) const {
        return __half2float(a) + b;
    }
    
    __device__ __forceinline__ double operator()(__half a, double b) const {
        return static_cast<double>(__half2float(a)) + b;
    }
     
    
    __device__ __forceinline__ float operator()(float a, __half b) const {
        return a + __half2float(b);
    }
    
    __device__ __forceinline__ double operator()(double a, __half b) const {
        return a + static_cast<double>(__half2float(b));
    }
    
    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, __half> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    __device__ __forceinline__ float operator()(__half a, B b) const {
        return __half2float(a) + static_cast<float>(b);
    }
    
    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, __half> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    __device__ __forceinline__ float operator()(A a, __half b) const {
        return static_cast<float>(a) + __half2float(b);
    }
     
};

struct Sub {
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A a, B b) const {
        return a - b;
    }

    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __hsub(a, b);
    }
    
    __device__ __forceinline__ float operator()(__half a, float b) const {
        return __half2float(a) - b;
    }
    
    __device__ __forceinline__ double operator()(__half a, double b) const {
        return static_cast<double>(__half2float(a)) - b;
    }
     
    __device__ __forceinline__ float operator()(float a, __half b) const {
        return a - __half2float(b);
    }
    
    __device__ __forceinline__ double operator()(double a, __half b) const {
        return a - static_cast<double>(__half2float(b));
    }
    
    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, __half> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    __device__ __forceinline__ float operator()(__half a, B b) const {
        return __half2float(a) - static_cast<float>(b);
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, __half> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    __device__ __forceinline__ float operator()(A a, __half b) const {
        return static_cast<float>(a) - __half2float(b);
    }
};

struct Mul { 

    template<class A, class B>
    __device__ __forceinline__ auto operator()(A a, B b) const {
        return a * b;
    }

    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __hmul(a, b);
    }
    
    __device__ __forceinline__ float operator()(__half a, float b) const {
        return __half2float(a) * b;
    }
    
    __device__ __forceinline__ double operator()(__half a, double b) const {
        return static_cast<double>(__half2float(a)) * b;
    } 
    
    __device__ __forceinline__ float operator()(float a, __half b) const {
        return a * __half2float(b);
    }
    
    __device__ __forceinline__ double operator()(double a, __half b) const {
        return a * static_cast<double>(__half2float(b));
    }
    
    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, __half> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    __device__ __forceinline__ float operator()(__half a, B b) const {
        return __half2float(a) * static_cast<float>(b);
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, __half> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    __device__ __forceinline__ float operator()(A a, __half b) const {
        return static_cast<float>(a) * __half2float(b);
    } 
};

struct Pow {  
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A a, B b) const {
        return pow(a, b);
    }

    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(powf(__half2float(a), __half2float(b)));
    }
    
    __device__ __forceinline__ float operator()(__half a, float b) const {
        return powf(__half2float(a), b);
    }
    
    __device__ __forceinline__ double operator()(__half a, double b) const {
        return pow(static_cast<double>(__half2float(a)), b);
    } 
    
    __device__ __forceinline__ float operator()(float a, __half b) const {
        return powf(a, __half2float(b));
    }
    
    __device__ __forceinline__ double operator()(double a, __half b) const {
        return pow(a, static_cast<double>(__half2float(b)));
    }
    
    template<typename B, typename = std::enable_if_t<!std::is_same_v<B, __half> && !std::is_same_v<B, float> && !std::is_same_v<B, double>>>
    __device__ __forceinline__ float operator()(__half a, B b) const {
        return powf(__half2float(a), static_cast<float>(b));
    }

    template<typename A, typename = std::enable_if_t<!std::is_same_v<A, __half> && !std::is_same_v<A, float> && !std::is_same_v<A, double>>>
    __device__ __forceinline__ float operator()(A a, __half b) const {
        return powf(static_cast<float>(a), __half2float(b));
    } 
    
     
};

constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}   
      
using BinaryOpKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);         
  
constexpr auto dispatchAdd = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Add>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Add>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Add>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Add>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Add>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Add>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Add>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Add>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Add>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Add>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Add>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Add>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Add>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Add>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Add>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Add>;

    table[index(float16, float16)] = launchBinaryOpKernel<__half, __half, __half, Add>;
    table[index(float16, float32)] = launchBinaryOpKernel<__half, float, float, Add>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, __half, float, Add>;
    table[index(float16, float64)] = launchBinaryOpKernel<__half, double, double, Add>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, __half, double, Add>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<__half, int8_t, float, Add>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, __half, float, Add>;
    table[index(float16, int16)]   = launchBinaryOpKernel<__half, int16_t, float, Add>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, __half, float, Add>;
    table[index(float16, int32)]   = launchBinaryOpKernel<__half, int32_t, float, Add>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, __half, float, Add>;
    table[index(float16, int64)]   = launchBinaryOpKernel<__half, int64_t, double, Add>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, __half, double, Add>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Add>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Add>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Add>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Add>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Add>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Add>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Add>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Add>;
    
    table[index(complex64, complex64)] = launchBinaryOpKernel<thrust::complex<float>, thrust::complex<float>, thrust::complex<float>, Add>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<thrust::complex<double>, thrust::complex<double>, thrust::complex<double>, Add>;
    return table;
}();  

constexpr auto dispatchSub = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Sub>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Sub>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Sub>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Sub>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Sub>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Sub>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Sub>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Sub>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Sub>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Sub>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Sub>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Sub>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Sub>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Sub>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Sub>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Sub>;

    table[index(float16, float16)] = launchBinaryOpKernel<__half, __half, __half, Sub>;
    table[index(float16, float32)] = launchBinaryOpKernel<__half, float, float, Sub>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, __half, float, Sub>;
    table[index(float16, float64)] = launchBinaryOpKernel<__half, double, double, Sub>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, __half, double, Sub>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<__half, int8_t, float, Sub>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, __half, float, Sub>;
    table[index(float16, int16)]   = launchBinaryOpKernel<__half, int16_t, float, Sub>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, __half, float, Sub>;
    table[index(float16, int32)]   = launchBinaryOpKernel<__half, int32_t, float, Sub>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, __half, float, Sub>;
    table[index(float16, int64)]   = launchBinaryOpKernel<__half, int64_t, double, Sub>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, __half, double, Sub>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Sub>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Sub>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Sub>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Sub>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Sub>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Sub>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Sub>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Sub>;


    table[index(complex64, complex64)] = launchBinaryOpKernel<thrust::complex<float>, thrust::complex<float>, thrust::complex<float>, Sub>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<thrust::complex<double>, thrust::complex<double>, thrust::complex<double>, Sub>;
    return table;
}(); 

constexpr auto dispatchMul = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Mul>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Mul>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Mul>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Mul>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Mul>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Mul>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Mul>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Mul>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Mul>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Mul>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Mul>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Mul>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Mul>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Mul>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Mul>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Mul>;

    table[index(float16, float16)] = launchBinaryOpKernel<__half, __half, __half, Mul>;
    table[index(float16, float32)] = launchBinaryOpKernel<__half, float, float, Mul>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, __half, float, Mul>;
    table[index(float16, float64)] = launchBinaryOpKernel<__half, double, double, Mul>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, __half, double, Mul>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<__half, int8_t, float, Mul>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, __half, float, Mul>;
    table[index(float16, int16)]   = launchBinaryOpKernel<__half, int16_t, float, Mul>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, __half, float, Mul>;
    table[index(float16, int32)]   = launchBinaryOpKernel<__half, int32_t, float, Mul>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, __half, float, Mul>;
    table[index(float16, int64)]   = launchBinaryOpKernel<__half, int64_t, double, Mul>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, __half, double, Mul>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Mul>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Mul>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Mul>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Mul>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Mul>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Mul>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Mul>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Mul>;

    table[index(complex64, complex64)] = launchBinaryOpKernel<thrust::complex<float>, thrust::complex<float>, thrust::complex<float>, Mul>;
    table[index(complex128, complex128)] = launchBinaryOpKernel<thrust::complex<double>, thrust::complex<double>, thrust::complex<double>, Mul>;
    return table;
}();   

constexpr auto dispatchPow = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel); 

    table[index(float16, float16)] = launchBinaryOpKernel<__half, __half, float, Pow>;
    table[index(float16, float32)] = launchBinaryOpKernel<__half, float, float, Pow>;
    table[index(float32, float16)] = launchBinaryOpKernel<float, __half, float, Pow>;
    table[index(float16, float64)] = launchBinaryOpKernel<__half, double, double, Pow>;
    table[index(float64, float16)] = launchBinaryOpKernel<double, __half, double, Pow>;
    
    table[index(float16, int8)]    = launchBinaryOpKernel<__half, int8_t, float, Pow>;
    table[index(int8, float16)]    = launchBinaryOpKernel<int8_t, __half, float, Pow>;
    table[index(float16, int16)]   = launchBinaryOpKernel<__half, int16_t, float, Pow>;
    table[index(int16, float16)]   = launchBinaryOpKernel<int16_t, __half, float, Pow>;
    table[index(float16, int32)]   = launchBinaryOpKernel<__half, int32_t, float, Pow>;
    table[index(int32, float16)]   = launchBinaryOpKernel<int32_t, __half, float, Pow>;
    table[index(float16, int64)]   = launchBinaryOpKernel<__half, int64_t, double, Pow>;
    table[index(int64, float16)]   = launchBinaryOpKernel<int64_t, __half, double, Pow>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Pow>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Pow>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Pow>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Pow>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Pow>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Pow>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Pow>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Pow>; 
    return table;
}();

} namespace cuda { 

status add(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchAdd[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
}

status sub(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchSub[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
} 

status mul(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchMul[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
}  

status pow(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchPow[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
}   

} // namespace cuda