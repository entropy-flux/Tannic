#include "cpu/kernels.h"

namespace cpu::matmul { 
template <typename L, typename R, typename O>
void matmul_nontransposed(const L* A, const R* B, O* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            O sum = O(0);
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

template <typename L, typename R, typename O>
void matmul_transposed(const L* A, const R* B, O* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            O sum = O(0);
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];   
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_float32_float32(const void* lhs, const void* rhs, void* out, size_t M, size_t N, size_t K, bool rhs_transposed) {
    if (rhs_transposed) {
        matmul_transposed(
            static_cast<const float*>(lhs),
            static_cast<const float*>(rhs),
            static_cast<float*>(out),
            M, N, K
        );
    } else {
        matmul_nontransposed(
            static_cast<const float*>(lhs),
            static_cast<const float*>(rhs),
            static_cast<float*>(out),
            M, N, K
        );
    }
}

void matmul_float32_float64(const void* lhs, const void* rhs, void* out, size_t M, size_t N, size_t K, bool rhs_transposed) {
    if (rhs_transposed) {
        matmul_transposed(
            static_cast<const float*>(lhs),
            static_cast<const double*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    } else {
        matmul_nontransposed(
            static_cast<const float*>(lhs),
            static_cast<const double*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    }
}

void matmul_float64_float32(const void* lhs, const void* rhs, void* out,
                            size_t M, size_t N, size_t K, bool rhs_transposed) {
    if (rhs_transposed) {
        matmul_transposed(
            static_cast<const double*>(lhs),
            static_cast<const float*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    } else {
        matmul_nontransposed(
            static_cast<const double*>(lhs),
            static_cast<const float*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    }
}

void matmul_float64_float64(const void* lhs, const void* rhs, void* out,
                            size_t M, size_t N, size_t K, bool rhs_transposed) {
    if (rhs_transposed) {
        matmul_transposed(
            static_cast<const double*>(lhs),
            static_cast<const double*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    } else {
        matmul_nontransposed(
            static_cast<const double*>(lhs),
            static_cast<const double*>(rhs),
            static_cast<double*>(out),
            M, N, K
        );
    }
}

}  // namespace cpu::matmul
