#include <cstddef>
#include <vector> 
#include <stdexcept>
#include <array>
#include "cpu/cpu.hpp" 
#include "Types.hpp"
 
template<typename S0, typename S1, typename D>
void gemmKernel( 
    bool transA, bool transB,
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int M, int N, int K,
    int lda, int ldb, int ldc
) { 
    const S0* A = static_cast<const S0*>(A_ptr);
    const S1* B = static_cast<const S1*>(B_ptr);
    D* C = static_cast<D*>(C_ptr);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            D sum = D(0);
            for (int k = 0; k < K; ++k) {
                size_t a_idx = transA ? k * lda + i : i * lda + k;
                size_t b_idx = transB ? j * ldb + k : k * ldb + j;
                sum += static_cast<D>(A[a_idx]) * static_cast<D>(B[b_idx]);
            }
            C[i * ldc + j] = sum;
        }
    }
}
 
#ifdef BLAS
#include <cblas.h> 

template <>
void gemmKernel<float, float, float>(
    bool transA, bool transB, 
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {  
    const float* A = static_cast<const float*>(A_ptr);
    const float* B = static_cast<const float*>(B_ptr);
    float* C = static_cast<float*>(C_ptr);

    cblas_sgemm(
        CblasRowMajor, 
        transA ? CblasTrans : CblasNoTrans, 
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.0, A, lda, B, ldb,
        0.0, C, ldc
    );
}

template <>
void gemmKernel<double, double, double>(
    bool transA, bool transB,
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int M, int N, int K,
    int lda, int ldb, int ldc
) { 
    const double* A = static_cast<const double*>(A_ptr);
    const double* B = static_cast<const double*>(B_ptr);
    double* C = static_cast<double*>(C_ptr);
    cblas_dgemm(
        CblasRowMajor, 
        transA ? CblasTrans : CblasNoTrans, 
        transB ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.0, A, lda, B, ldb,
        0.0, C, ldc
    );
} 
#endif
  
using Kernel = void(*)(
    bool transA, bool transB,
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int M, int N, int K,
    int lda, int ldb, int ldc
);       

constexpr void defaultKernel(
    bool transA, bool transB,
    const void* A_ptr,
    const void* B_ptr,
    void* C_ptr,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    throw std::runtime_error("Not supported dtype");
};
 
constexpr static inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
} 

constexpr auto gemm = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(defaultKernel); 
    table[index(int8, int8)]     = gemmKernel<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]    = gemmKernel<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]    = gemmKernel<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]    = gemmKernel<int8_t, int64_t, int64_t>;

    table[index(int16, int8)]    = gemmKernel<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]   = gemmKernel<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]   = gemmKernel<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]   = gemmKernel<int16_t, int64_t, int64_t>;

    table[index(int32, int8)]    = gemmKernel<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]   = gemmKernel<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]   = gemmKernel<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]   = gemmKernel<int32_t, int64_t, int64_t>;

    table[index(int64, int8)]    = gemmKernel<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]   = gemmKernel<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]   = gemmKernel<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]   = gemmKernel<int64_t, int64_t, int64_t>;

    table[index(int32, float32)] = gemmKernel<int32_t, float, float>;
    table[index(float32, int32)] = gemmKernel<float, int32_t, float>;
    table[index(int32, float64)] = gemmKernel<int32_t, double, double>;
    table[index(float64, int32)] = gemmKernel<double, int32_t, double>;

    table[index(float32, float32)] = gemmKernel<float, float, float>;
    table[index(float32, float64)] = gemmKernel<float, double, double>;
    table[index(float64, float32)] = gemmKernel<double, float, double>;
    table[index(float64, float64)] = gemmKernel<double, double, double>;
    return table;
}();

namespace cpu {

using tannic::dsizeof;

void gemm(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) { 
    size_t batch_rank = dst->rank > 2 ? dst->rank - 2 : 0;
    size_t batch_size = 1;
    for (size_t dim = 0; dim < batch_rank; ++dim) {
        batch_size *= dst->shape[dim];
    }

    size_t M = dst->shape[dst->rank - 2];
    size_t N = dst->shape[dst->rank - 1];
    size_t K = src0->shape[src0->rank-1];
 
    auto unravel = [&](size_t b, size_t* idxs){
        for (int i = batch_rank-1, r=b; i >= 0; --i) {
            idxs[i] = r % dst->shape[i];
            r /= dst->shape[i];
        }
    }; 
    
    std::vector<size_t> dst_idx(batch_rank), s0_idx(batch_rank), s1_idx(batch_rank); 

    for (size_t batch = 0; batch < batch_size; ++batch) {
        if (batch_rank)
            unravel(batch, dst_idx.data());


        for (size_t dim = 0; dim < batch_rank; ++dim) {
            s0_idx[dim] = (src0->rank > 2 && src0->shape[dim] == 1) ? 0 : dst_idx[dim];
            s1_idx[dim] = (src1->rank > 2 && src1->shape[dim] == 1) ? 0 : dst_idx[dim];
        } 

        size_t offs_src0 = 0, offs_src1 = 0, offs_dst = 0;

        for (size_t dim = 0; dim < batch_rank; ++dim) {
            offs_src0 += s0_idx[dim] * src0->strides[dim];
            offs_src1 += s1_idx[dim] * src1->strides[dim];
            offs_dst += dst_idx[dim] * dst->strides[dim];
        }  

        int lda = src0_transposed ? src0->strides[src0->rank-1] : src0->strides[src0->rank-2];
        int ldb = src1_transposed ? src1->strides[src1->rank-1] : src1->strides[src1->rank-2];
        int ldc = dst->strides[dst->rank-2];

        const void* A = static_cast<const char*>(src0->address) + offs_src0 * dsizeof(src0->dtype); 
        const void* B = static_cast<const char*>(src1->address) + offs_src1 * dsizeof(src0->dtype); 
        void* C = static_cast<char*>(dst->address) + offs_dst * dsizeof(src0->dtype);
 
        ::gemm[index(src0->dtype, src1->dtype)](
            src0_transposed, src1_transposed,
            A, B, C,
            M, N, K,
            lda,
            ldb, 
            ldc
        );
    }  
} 

} // namespace cpu