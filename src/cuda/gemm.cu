#include <vector>
#include <array>
#include <stdexcept>
#include "cuda/streams.cuh"
#include "cuda/gemm.cuh"

template<typename S0, typename S1, typename D>
__global__ void gemmKernel(
    bool A_trans, bool B_trans,
    const S0* __restrict__ A_ptr,
    const S1* __restrict__ B_ptr,
    D* __restrict__ C_ptr,
    size_t M, size_t N, size_t K,
    size_t A_ld, size_t B_ld, size_t C_ld  
) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;
    D sum = D(0);
    for (size_t k = 0; k < K; ++k) {
        size_t A_idx = A_trans ? k * A_ld + i : i * A_ld + k;
        size_t B_idx = B_trans ? j * B_ld + k : k * B_ld + j;
        sum += static_cast<D>(A_ptr[A_idx]) * static_cast<D>(B_ptr[B_idx]);
    }   
    C_ptr[i * C_ld + j] = sum;
}

static bool isTransposed(const tensor_t* tensor) {
    if(tensor->rank < 2) {
        return false;
    }
    else {
        return tensor->strides.sizes[tensor->rank-1] > tensor->strides.sizes[tensor->rank-2];
    }
}   

template<typename S0, typename S1, typename D>
void computeOffsets(
    const tensor_t* src0, const tensor_t* src1, const tensor_t* dst,
    size_t* dst_idx,
    size_t& offs_src0, size_t& offs_src1, size_t& offs_dst,
    uint8_t batch, uint8_t batch_rank
) {
 
    size_t b = batch;
    for (int dim = batch_rank - 1; dim >= 0; --dim) {
        dst_idx[dim] = b % dst->shape.sizes[dim];
        b /= dst->shape.sizes[dim];
    } 

    for (size_t dim = 0; dim < batch_rank; ++dim) {
        size_t s0_idx = (src0->rank > 2 && src0->shape.sizes[dim] == 1) ? 0 : dst_idx[dim];
        size_t s1_idx = (src1->rank > 2 && src1->shape.sizes[dim] == 1) ? 0 : dst_idx[dim];

        offs_src0 += s0_idx * src0->strides.sizes[dim];
        offs_src1 += s1_idx * src1->strides.sizes[dim];
        offs_dst  += dst_idx[dim] * dst->strides.sizes[dim];
    }
}



template<typename S0, typename S1, typename D>
status launchGemmKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {  
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t M = dst->shape.sizes[dst->rank - 2];
    size_t N = dst->shape.sizes[dst->rank - 1];
    size_t K = src0->shape.sizes[src0->rank - 1];

    bool A_trans = isTransposed(src0);
    bool B_trans = isTransposed(src1);

    int A_ld = A_trans ? src0->strides.sizes[src0->rank - 1] : src0->strides.sizes[src0->rank - 2];
    int B_ld = B_trans ? src1->strides.sizes[src1->rank - 1] : src1->strides.sizes[src1->rank - 2];
    int C_ld = dst->strides.sizes[dst->rank - 2];
  
    if (dst->rank > 2) { 
        size_t dst_idx[6] = {0};
        uint8_t batch_rank = dst->rank - 2;  
        size_t batch_size = 1;
        for (int dim = 0; dim < batch_rank; ++dim)
            batch_size *= dst->shape.sizes[dim];

        for (size_t batch = 0; batch < batch_size; ++batch) {
            size_t offs_src0 = 0, offs_src1 = 0, offs_dst = 0;
            computeOffsets<S0, S1, D>(src0, src1, dst, dst_idx, offs_src0, offs_src1, offs_dst, batch, batch_rank);

            const S0* A_ptr = static_cast<const S0*>(src0->address) + offs_src0;
            const S1* B_ptr = static_cast<const S1*>(src1->address) + offs_src1;
            D* C_ptr = static_cast<D*>(dst->address) + offs_dst;

            dim3 blockDim(16, 16);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
            gemmKernel<S0, S1, D><<<gridDim, blockDim, 0, cudaStream>>>(
                A_trans, B_trans,
                A_ptr, B_ptr, C_ptr,
                M, N, K,
                A_ld, B_ld, C_ld
            );
        } 
    }

    else {
        const S0* A_ptr = static_cast<const S0*>(src0->address);
        const S1* B_ptr = static_cast<const S1*>(src1->address); 
        D* C_ptr = static_cast<D*>(dst->address);
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        gemmKernel<S0, S1, D><<<gridDim, blockDim, 0, cudaStream>>>(
            A_trans, B_trans,
            A_ptr, B_ptr, C_ptr,
            M, N, K,
            A_ld, B_ld, C_ld
        );
    }

    return SUCCESS;
}

status defaultKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t) {
    return UNSUPORTED_DTYPE;
};

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, stream_t);        
 
constexpr static inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
} 

constexpr auto dispatchGemm = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(defaultKernel); 
    table[index(int8, int8)]     = launchGemmKernel<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = launchGemmKernel<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = launchGemmKernel<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = launchGemmKernel<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = launchGemmKernel<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = launchGemmKernel<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = launchGemmKernel<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = launchGemmKernel<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = launchGemmKernel<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = launchGemmKernel<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = launchGemmKernel<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = launchGemmKernel<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = launchGemmKernel<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = launchGemmKernel<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = launchGemmKernel<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = launchGemmKernel<int64_t, int64_t, int64_t>;

    table[index(int32, float32)] = launchGemmKernel<int32_t, float, float>;
    table[index(float32, int32)] = launchGemmKernel<float, int32_t, float>;
    table[index(int32, float64)] = launchGemmKernel<int32_t, double, double>;
    table[index(float64, int32)] = launchGemmKernel<double, int32_t, double>;

    table[index(float32, float32)] = launchGemmKernel<float, float, float>;
    table[index(float32, float64)] = launchGemmKernel<float, double, double>;
    table[index(float64, float32)] = launchGemmKernel<double, float, double>;
    table[index(float64, float64)] = launchGemmKernel<double, double, double>;
    return table;
}();

namespace cuda { 

status gemm(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, stream_t stream) {    
    return dispatchGemm[index(src0->dtype, src1->dtype)](src0, src1, dst, stream);  
} 

} 