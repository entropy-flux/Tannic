#include <cstddef>
#include <vector>

#include "cpu/matmul-op.hpp"

namespace cpu {

template <typename S, typename D, typename TC>
void gemm(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) {
    const S* src0_data = reinterpret_cast<const S*>(src0->address) + src0->offset;
    const D* src1_data = reinterpret_cast<const D*>(src1->address) + src1->offset;
    TC* dst_data = reinterpret_cast<TC*>(dst->address) + dst->offset;

    const size_t M = dst->shape[dst->rank - 2];
    const size_t N = dst->shape[dst->rank - 1];
    const size_t K = src0_transposed ? src0->shape[src0->rank - 2] : src0->shape[src0->rank - 1];

    const size_t* s0_strides = src0->strides;
    const size_t* s1_strides = src1->strides;
    const size_t* d_strides = dst->strides;

    const size_t s0_rank = src0->rank;
    const size_t s1_rank = src1->rank;
    const size_t d_rank  = dst->rank;

    if (src0_transposed && src1_transposed) { 
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t s0_offset = k * s0_strides[s0_rank - 2] + m * s0_strides[s0_rank - 1];
                    size_t s1_offset = n * s1_strides[s1_rank - 2] + k * s1_strides[s1_rank - 1];
                    sum += static_cast<TC>(src0_data[s0_offset]) * static_cast<TC>(src1_data[s1_offset]);
                }
                size_t d_offset = m * d_strides[d_rank - 2] + n * d_strides[d_rank - 1];
                dst_data[d_offset] = sum;
            }
        }
    } else if (src0_transposed && !src1_transposed) { 
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t s0_offset = k * s0_strides[s0_rank - 2] + m * s0_strides[s0_rank - 1];
                    size_t s1_offset = k * s1_strides[s1_rank - 2] + n * s1_strides[s1_rank - 1];
                    sum += static_cast<TC>(src0_data[s0_offset]) * static_cast<TC>(src1_data[s1_offset]);
                }
                size_t d_offset = m * d_strides[d_rank - 2] + n * d_strides[d_rank - 1];
                dst_data[d_offset] = sum;
            }
        }
    } else if (!src0_transposed && src1_transposed) { 
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t s0_offset = m * s0_strides[s0_rank - 2] + k * s0_strides[s0_rank - 1];
                    size_t s1_offset = n * s1_strides[s1_rank - 2] + k * s1_strides[s1_rank - 1];
                    sum += static_cast<TC>(src0_data[s0_offset]) * static_cast<TC>(src1_data[s1_offset]);
                }
                size_t d_offset = m * d_strides[d_rank - 2] + n * d_strides[d_rank - 1];
                dst_data[d_offset] = sum;
            }
        }
    } else { 
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                TC sum = TC(0);
                for (size_t k = 0; k < K; ++k) {
                    size_t s0_offset = m * s0_strides[s0_rank - 2] + k * s0_strides[s0_rank - 1];
                    size_t s1_offset = k * s1_strides[s1_rank - 2] + n * s1_strides[s1_rank - 1];
                    sum += static_cast<TC>(src0_data[s0_offset]) * static_cast<TC>(src1_data[s1_offset]);
                }
                size_t d_offset = m * d_strides[d_rank - 2] + n * d_strides[d_rank - 1];
                dst_data[d_offset] = sum;
            }
        }
    }
} 
 

#ifdef BLAS

#include <cblas.h>

template <>
void gemm<float, float, float>(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) {
    const float* A = reinterpret_cast<const float*>(src0->address) + src0->offset;
    const float* B = reinterpret_cast<const float*>(src1->address) + src1->offset;
    float* C = reinterpret_cast<float*>(dst->address) + dst->offset;

    const int M = static_cast<int>(dst->shape[dst->rank - 2]);
    const int N = static_cast<int>(dst->shape[dst->rank - 1]);
    const int K = static_cast<int>(src0_transposed ? src0->shape[src0->rank - 2] : src0->shape[src0->rank - 1]);

    const CBLAS_TRANSPOSE transA = src0_transposed ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transB = src1_transposed ? CblasTrans : CblasNoTrans;

    const int lda = static_cast<int>(src0->strides[src0->rank - 2]);
    const int ldb = static_cast<int>(src1->strides[src1->rank - 2]);
    const int ldc = static_cast<int>(dst->strides[dst->rank - 2]);

    cblas_sgemm(
        CblasRowMajor, 
        transA, transB,       
        M, N, K,
        1.0f, A, lda, B, ldb,
        0.0f, C, ldc
    );
}
 
template <>
void gemm<double, double, double>(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) {
    const double* A = reinterpret_cast<const double*>(src0->address) + src0->offset;
    const double* B = reinterpret_cast<const double*>(src1->address) + src1->offset;
    double* C = reinterpret_cast<double*>(dst->address) + dst->offset;

    const int M = static_cast<int>(dst->shape[dst->rank - 2]);
    const int N = static_cast<int>(dst->shape[dst->rank - 1]);
    const int K = static_cast<int>(src0_transposed ? src0->shape[src0->rank - 2] : src0->shape[src0->rank - 1]);

    const CBLAS_TRANSPOSE transA = src0_transposed ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transB = src1_transposed ? CblasTrans : CblasNoTrans;

    const int lda = static_cast<int>(src0->strides[src0->rank - 2]);
    const int ldb = static_cast<int>(src1->strides[src1->rank - 2]);
    const int ldc = static_cast<int>(dst->strides[dst->rank - 2]);

    cblas_dgemm(
        CblasRowMajor, transA, transB,
        M, N, K,
        1.0, A, lda, B, ldb,
        0.0, C, ldc
    );
}

#endif 

inline void unravel_index(size_t flat_idx, const size_t* shape, size_t rank, size_t* indices) {
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
        indices[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
}

inline size_t compute_batch_offset(const tensor_t* t, const size_t* indices, size_t batch_rank) {
    size_t offset = 0;
    for (size_t i = 0; i < batch_rank; ++i) {
        if (t->rank > 2) {
            offset += (t->shape[i] == 1 ? 0 : indices[i]) * t->strides[i];
        }
    }
    return offset;
}

template <typename S, typename D, typename TC>
void matmul_op(const tensor_t* src0, const tensor_t* src1, tensor_t* dst, bool src0_transposed, bool src1_transposed) {
    const size_t batch_rank = (dst->rank > 2) ? dst->rank - 2 : 0;

    size_t batch_size = 1;
    for (size_t i = 0; i < batch_rank; ++i)
        batch_size *= dst->shape[i];

    std::vector<size_t> dst_indices(batch_rank);
    std::vector<size_t> src0_indices(batch_rank);
    std::vector<size_t> src1_indices(batch_rank);

    for (size_t b = 0; b < batch_size; ++b) {
        if (batch_rank > 0)
            unravel_index(b, dst->shape, batch_rank, dst_indices.data());

        for (size_t i = 0; i < batch_rank; ++i) {
            src0_indices[i] = (src0->rank > 2 && src0->shape[i] == 1) ? 0 : dst_indices[i];
            src1_indices[i] = (src1->rank > 2 && src1->shape[i] == 1) ? 0 : dst_indices[i];
        }

        tensor_t src0_batch = *src0;
        tensor_t src1_batch = *src1;
        tensor_t dst_batch  = *dst;

        src0_batch.offset += compute_batch_offset(src0, src0_indices.data(), batch_rank);
        src1_batch.offset += compute_batch_offset(src1, src1_indices.data(), batch_rank);
        dst_batch.offset  += compute_batch_offset(dst,  dst_indices.data(),  batch_rank);

        gemm<S, D, TC>(&src0_batch, &src1_batch, &dst_batch, src0_transposed, src1_transposed);
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