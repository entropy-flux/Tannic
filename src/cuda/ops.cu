#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/ops.cuh"
#include "cuda/streams.cuh"

template<typename S, typename D, class Op>
__global__ void scalarUnaryOpKernel(const S* src, D* dst) {
    Op op;
    *dst = op(*src);
}  

template<typename S, typename D, class Op>
__global__ void batchedUnaryOpKernel(
    const S* src, shape_t src_shape, strides_t src_strides,           
    D* dst, shape_t dst_shape, strides_t dst_strides, 
    uint8_t rank, size_t ne
) {
    Op op{};

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) { 
        size_t offs = 0;
        size_t remaining = idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % dst_shape.sizes[dim];
            remaining /= dst_shape.sizes[dim];
 
            size_t src_idx = (src_shape.sizes[dim] == 1) ? 0 : dim_idx;
            offs += src_idx * src_strides.sizes[dim];
        }

        dst[idx] = op(src[offs]);
    }
} 

template<typename S, typename D, class Op>
status launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, cudaStream_t stream = 0) { 
    if (src->rank == 0) {
        scalarUnaryOpKernel<S, D, Op><<<1, 1, 0,stream>>>(
            (const S*)(src->address),
            (D*)(dst->address)
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedUnaryOpKernel<S, D, Op><<<gridSize, blockSize, 0, stream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
    return SUCCESS;
} 

template<typename S0, typename S1, typename D, class Op>
__global__ void scalarBinaryOpKernel(const S0* src0, const S1* src1, D* dst) {
    Op op;
    *dst = op(*src0, *src1);
}


template<typename S0, typename S1, typename D, class Op>
__global__ void batchedBinaryOpKernel(
    const S0* src0_ptr, shape_t src0_shape, strides_t src0_strides,
    const S1* src1_ptr, shape_t src1_shape, strides_t src1_strides,
    D* dst_ptr, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne
) {
    Op op{};  
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne;  idx += blockDim.x * gridDim.x) {
        size_t cnt[8] = {0};
        size_t remaining = idx;
         
        for (uint8_t i = rank - 1; i > 0; --i) {
            cnt[i] = remaining % dst_shape.sizes[i];
            remaining /= dst_shape.sizes[i];
        }
        cnt[0] = remaining;
         
        size_t offs0 = 0, offs1 = 0;
        for (uint8_t i = 0; i < rank; ++i) {
            size_t idx0 = (src0_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx1 = (src1_shape.sizes[i] == 1) ? 0 : cnt[i];
            
            offs0 += idx0 * src0_strides.sizes[i];
            offs1 += idx1 * src1_strides.sizes[i];
        }
         
        dst_ptr[idx] = op(src0_ptr[offs0], src1_ptr[offs1]);
    }
}
  
 
template<typename S, typename D, class Op>
status launchUnaryOpKernel(const tensor_t* src, tensor_t* dst, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    if (src->rank == 0) {
        scalarUnaryOpKernel<S, D, Op><<<1, 1, 0, cudaStream>>>(
            (const S*)(src->address), 
            (D*)(dst->address)
        ); 
    } 
    
    else {    
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedUnaryOpKernel<S, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
    return SUCCESS;
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
        size_t ne = 1;
        for (uint8_t dim = 0; dim < dst->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }
        
        int blockSize = 256;
        int gridSize = (ne + blockSize - 1) / blockSize;

        batchedBinaryOpKernel<S0, S1, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S0*)(src0->address), src0->shape, src0->strides,
            (const S1*)(src1->address), src1->shape, src1->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            dst->rank, ne
        ); 
    } 
    return SUCCESS;
}         


status launchDefaultUnaryOpKernel(const tensor_t*, tensor_t*, stream_t) {
    return UNSUPORTED_DTYPE;
};  
  

status launchDefaultBinaryOpKernel(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {
    return UNSUPORTED_DTYPE;
};   

struct Neg { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(-a)) {
        return -a;
    }
};

struct Add { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a + b)) {
        return a + b;
    }
};

struct Sub { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a - b)) {
        return a - b;
    }
};

struct Mul { 
    template<class A, class B>
    __device__ __forceinline__ auto operator()(A&& a, B&& b) const noexcept(noexcept(a * b)) {
        return a * b;
    }
};   

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}


constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}   

using UnaryOpKernel = status(*)(const tensor_t*, tensor_t*, stream_t);       
using BinaryOpKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);         

constexpr auto dispatchNeg = []() {  
    std::array<UnaryOpKernel, index(TYPES)> table; table.fill(launchDefaultUnaryOpKernel);
    table[index(int8)] = launchUnaryOpKernel<int8_t, int8_t, Neg>;
    table[index(int16)] = launchUnaryOpKernel<int16_t, int16_t, Neg>;
    table[index(int32)] = launchUnaryOpKernel<int32_t, int32_t, Neg>;
    table[index(int64)] = launchUnaryOpKernel<int64_t, int64_t, Neg>;
    table[index(float32)] = launchUnaryOpKernel<float, float, Neg>;
    table[index(float64)] = launchUnaryOpKernel<double, double, Neg>;
    return table;
}();      

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
    return table;
}();  

namespace cuda {

status neg(tensor_t const* src, tensor_t* dst, stream_t stream) {   
    return dispatchNeg[index(src->dtype)](src, dst, stream); 
}

status add(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchAdd[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
}

status sub(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchSub[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
} 

status mul(tensor_t const* src1, tensor_t const* src2, tensor_t* dst, stream_t stream) {  
    return dispatchMul[index(src1->dtype, src2->dtype)](src1, src2, dst, stream); 
}  

} // namespace cuda