#include <cstddef>
#include <cstdint>
#include <cstring>
#include "Types.hpp"

namespace cpu::linear {

template <typename TA, typename TB, typename TC>
void matmul_impl(
    bool is_rowMajor,
    bool A_is_transposed,
    bool B_is_transposed,
    int M, int N, int K,
    const void* A_, int lda,
    const void* B_, int ldb,
    void* C_, int ldc
) {
    const TA* A = static_cast<const TA*>(A_);
    const TB* B = static_cast<const TB*>(B_);
    TC* C = static_cast<TC*>(C_);

    if (is_rowMajor) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                TC sum = 0;
                for (int k = 0; k < K; ++k) {
                    TA a_val = A_is_transposed ? A[k * lda + i] : A[i * lda + k];
                    TB b_val = B_is_transposed ? B[j * ldb + k] : B[k * ldb + j];
                    sum += static_cast<TC>(a_val) * static_cast<TC>(b_val);
                }
                C[i * ldc + j] = sum;
            }
        }
    } else {
        // Column-major layout
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                TC sum = 0;
                for (int k = 0; k < K; ++k) {
                    TA a_val = A_is_transposed ? A[i + k * lda] : A[k + i * lda];
                    TB b_val = B_is_transposed ? B[k + j * ldb] : B[j + k * ldb];
                    sum += static_cast<TC>(a_val) * static_cast<TC>(b_val);
                }
                C[i + j * ldc] = sum;
            }
        }
    }
}





void linear_float32_float32(
    bool is_rowMajor, bool A_is_transposed, bool B_is_transposed,
    int M, int N, int K,
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc
) {
    matmul_impl<float, float, float>(
        is_rowMajor, A_is_transposed, B_is_transposed,
        M, N, K, A, lda, B, ldb, C, ldc);
}

void linear_float32_float64(
    bool is_rowMajor, bool A_is_transposed, bool B_is_transposed,
    int M, int N, int K,
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc
) {
    matmul_impl<float, double, double>(
        is_rowMajor, A_is_transposed, B_is_transposed,
        M, N, K, A, lda, B, ldb, C, ldc);
}

void linear_float64_float32(
    bool is_rowMajor, bool A_is_transposed, bool B_is_transposed,
    int M, int N, int K,
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc
) {
    matmul_impl<double, float, double>(
        is_rowMajor, A_is_transposed, B_is_transposed,
        M, N, K, A, lda, B, ldb, C, ldc);
}

void linear_float64_float64(
    bool is_rowMajor, bool A_is_transposed, bool B_is_transposed,
    int M, int N, int K,
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc
) {
    matmul_impl<double, double, double>(
        is_rowMajor, A_is_transposed, B_is_transposed,
        M, N, K, A, lda, B, ldb, C, ldc);
}


} // namespace cpu::linear
