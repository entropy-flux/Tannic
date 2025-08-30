#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "cuda/exc.cuh"
#include "cuda/ops.cuh" 

namespace { 

template<typename S0, typename S1, typename D, class Op>
__global__ void scalarBinaryOpKernel(const S0* src0, const S1* src1, D* dst) {
    Op op;
    *dst = op(*src0, *src1);
} 

template<typename S0, typename S1, typename D, class Op>
__global__ void batchedBinaryOpKernel(
    const S0* src0_ptr, shape_t src0_shape, strides_t src0_strides, uint8_t src0_rank,
    const S1* src1_ptr, shape_t src1_shape, strides_t src1_strides, uint8_t src1_rank,
    D* dst_ptr, shape_t dst_shape, strides_t dst_strides, uint8_t dst_rank,
    size_t ne
) {
    Op op{};
    
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        size_t cnt[8] = {0};  
        size_t remaining = idx;
 
        for (int i = dst_rank - 1; i >= 0; --i) {
            cnt[i] = remaining % dst_shape.sizes[i];
            remaining /= dst_shape.sizes[i];
        }

        size_t offs0 = 0, offs1 = 0;
        for (uint8_t i = 0; i < dst_rank; ++i) { 
            int dim0 = i + src0_rank - dst_rank;
            int dim1 = i + src1_rank - dst_rank;

            size_t idx0 = (dim0 >= 0 && src0_shape.sizes[dim0] > 1) ? cnt[i] : 0;
            size_t idx1 = (dim1 >= 0 && src1_shape.sizes[dim1] > 1) ? cnt[i] : 0;

            if (dim0 >= 0) offs0 += idx0 * src0_strides.sizes[dim0];
            if (dim1 >= 0) offs1 += idx1 * src1_strides.sizes[dim1];
        }

        dst_ptr[idx] = op(src0_ptr[offs0], src1_ptr[offs1]);
    }
} 

template<typename S0, typename S1, typename D, class Op>
status launchBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);

    if (dst->rank == 0) {
        scalarBinaryOpKernel<S0, S1, D, Op><<<1, 1, 0, cudaStream>>>(
            (const S0*)(src0->address), 
            (const S1*)(src1->address), 
            (D*)(dst->address)
        );   
    } 
    else {     
        // total number of elements in dst
        size_t ne = dst->size; 
        int blockSize = 256;
        int gridSize = (ne + blockSize - 1) / blockSize;

        batchedBinaryOpKernel<S0, S1, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S0*)(src0->address), src0->shape, src0->strides, src0->rank,
            (const S1*)(src1->address), src1->shape, src1->strides, src1->rank,
            (D*)(dst->address), dst->shape, dst->strides, dst->rank,
            ne
        ); 
    } 

    return SUCCESS;
}  

constexpr static status launchDefaultBinaryOpKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {
    return UNSUPPORTED_DTYPE;
};    

struct Add { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const {
        return a + b;
    }
};

struct Sub { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const {
        return a - b;
    }
};

struct Mul { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const {
        return a * b;
    }
};    

struct Pow { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const {
        return pow(a, b);
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