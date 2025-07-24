#include <cstddef>
#include <vector> 
#include "cpu/gemm.hpp"
 

namespace cpu {
 
template<typename S0, typename S1, typename D>
void gemmKernel( 
    bool transA, bool transB,
    const S0* A,
    const S1* B,
    D* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) { 
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
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int lda,
    int ldb,
    int ldc
) {  
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
    const double* A,
    const double* B,
    double* C,
    int M, int N, int K,
    int lda,
    int ldb,
    int ldc
) { 
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
 

template<typename S0, typename S1, typename D>
bool gemm(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) { 
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

        gemmKernel<S0,S1,D>(
            src0_transposed, src1_transposed,
            static_cast<const S0*>(src0->address) + offs_src0,
            static_cast<const S1*>(src1->address) + offs_src1,
            reinterpret_cast<D*>(dst->address) + offs_dst,
            M, N, K,
            lda,
            ldb, 
            ldc
        );
    } 
    return true;
}


} // namespace cpu

template bool cpu::gemm<int8_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int8_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int8_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int8_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template bool cpu::gemm<int16_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int16_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int16_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int16_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template bool cpu::gemm<int32_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int32_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int32_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int32_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template bool cpu::gemm<int64_t, int8_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int64_t, int16_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int64_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int64_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template bool cpu::gemm<float, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<float, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<double, float, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<double, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template bool cpu::gemm<int32_t, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<float, int32_t, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<int32_t, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template bool cpu::gemm<double, int32_t, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);