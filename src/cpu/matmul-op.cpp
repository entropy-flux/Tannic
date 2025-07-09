#include <cstddef>
#include <vector>

#include "cpu/matmul-op.hpp"

#include <cstddef>
#include <vector>

namespace cpu {

template <typename S, typename D, typename TC>
void gemmKernel(
    const S* A,
    const D* B,
    TC* C,
    size_t M, size_t N, size_t K,
    size_t lda0, size_t lda1,
    size_t ldb0, size_t ldb1,
    size_t ldc0, size_t ldc1,
    bool transA, bool transB
) {
    if (transA && transB) { 
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) { 
                    size_t a_idx = k * lda0 + m * lda1; 
                    size_t b_idx = n * ldb0 + k * ldb1;
                    sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
                }
                C[m * ldc0 + n * ldc1] = sum;
            }
        }
    }
    else if (transA && !transB) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t a_idx = k * lda0 + m * lda1;
                    size_t b_idx = k * ldb0 + n * ldb1;
                    sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
                }
                C[m * ldc0 + n * ldc1] = sum;
            }
        }
    } 
    else if (!transA && transB) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t a_idx = m * lda0 + k * lda1;
                    size_t b_idx = n * ldb0 + k * ldb1;
                    sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
                }
                C[m * ldc0 + n * ldc1] = sum;
            }
        }
    } 
    else {  
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t a_idx = m * lda0 + k * lda1;
                    size_t b_idx = k * ldb0 + n * ldb1;
                    sum += static_cast<TC>(A[a_idx]) * static_cast<TC>(B[b_idx]);
                }
                C[m * ldc0 + n * ldc1] = sum;
            }
        }
    }
}

#ifdef BLAS
#include <cblas.h>

template <>
void gemmKernel<float, float, float>(
    const float* A,
    const float* B,
    float* C,
    size_t M, size_t N, size_t K,
    size_t lda0, size_t lda1,
    size_t ldb0, size_t ldb1,
    size_t ldc0, size_t ldc1,
    bool transA, bool transB
) {
    const CBLAS_TRANSPOSE cblasTransA = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE cblasTransB = transB ? CblasTrans : CblasNoTrans;
    
    cblas_sgemm(
        CblasRowMajor, 
        cblasTransA, cblasTransB,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        1.0f, A, static_cast<int>(lda0), B, static_cast<int>(ldb0),
        0.0f, C, static_cast<int>(ldc0)
    );
}

template <>
void gemmKernel<double, double, double>(
    const double* A,
    const double* B,
    double* C,
    size_t M, size_t N, size_t K,
    size_t lda0, size_t lda1,
    size_t ldb0, size_t ldb1,
    size_t ldc0, size_t ldc1,
    bool transA, bool transB
) {
    const CBLAS_TRANSPOSE cblasTransA = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE cblasTransB = transB ? CblasTrans : CblasNoTrans;
    
    cblas_dgemm(
        CblasRowMajor, 
        cblasTransA, cblasTransB,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        1.0, A, static_cast<int>(lda0), B, static_cast<int>(ldb0),
        0.0, C, static_cast<int>(ldc0)
    );
}
#endif

template <typename S, typename D, typename TC>
void matmul_op(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool transA, bool transB) {
    const size_t batch_rank = (dst->rank > 2) ? dst->rank - 2 : 0;
    size_t batch_size = 1;
    for (size_t i = 0; i < batch_rank; ++i)
        batch_size *= dst->shape[i];

    size_t M = dst->shape[dst->rank - 2];
    size_t N = dst->shape[dst->rank - 1];
    size_t K = transA ? src0->shape[src0->rank - 2] : src0->shape[src0->rank - 1];

    auto unravel = [&](size_t b, size_t* idxs) {
        for (int i = static_cast<int>(batch_rank) - 1, r = b; i >= 0; --i) {
            idxs[i] = r % dst->shape[i];
            r /= dst->shape[i];
        }
    };

    std::vector<size_t> dst_idx(batch_rank), s0_idx(batch_rank), s1_idx(batch_rank);
    
    for (size_t b = 0; b < batch_size; ++b) {
        if (batch_rank > 0)
            unravel(b, dst_idx.data());

        for (size_t i = 0; i < batch_rank; ++i) {
            s0_idx[i] = (src0->rank > 2 && src0->shape[i] == 1) ? 0 : dst_idx[i];
            s1_idx[i] = (src1->rank > 2 && src1->shape[i] == 1) ? 0 : dst_idx[i];
        }

        size_t off0 = 0, off1 = 0, offD = 0;
        for (size_t i = 0; i < batch_rank; ++i) {
            off0 += s0_idx[i] * src0->strides[i];
            off1 += s1_idx[i] * src1->strides[i];
            offD += dst_idx[i] * dst->strides[i];
        }

        gemmKernel<S, D, TC>(
            reinterpret_cast<const S*>(src0->address) + src0->offset + off0,
            reinterpret_cast<const D*>(src1->address) + src1->offset + off1,
            reinterpret_cast<TC*>(dst->address) + dst->offset + offD,
            M, N, K,
            src0->strides[src0->rank - 2],
            src0->strides[src0->rank - 1],
            src1->strides[src1->rank - 2],
            src1->strides[src1->rank - 1],
            dst->strides[dst->rank - 2],
            dst->strides[dst->rank - 1],
            transA, transB
        );
    }
}

} // namespace cpu

// Explicit template instantiations 
template void cpu::matmul_op<int8_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int16_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int32_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int64_t, int8_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int16_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<float, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<float, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, float, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int32_t, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<float, int32_t, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, int32_t, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);