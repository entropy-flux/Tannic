#include <cblas.h> 
#include "kernels/openblas/gemm.h"

namespace openblas::gemm {

void sgemm(
    Order order, Transposed TransA, Transposed TransB,
    int M, int N, int K,
    double alpha, 
    const void* A, int lda,
    const void* B, int ldb,
    double beta,
    void* C, int ldc
) {
    cblas_sgemm(
        static_cast<CBLAS_ORDER>(order), 
        static_cast<CBLAS_TRANSPOSE>(TransA), static_cast<CBLAS_TRANSPOSE>(TransB), 
        M, N, K,
        static_cast<float>(alpha),
        static_cast<const float*>(A), lda,
        static_cast<const float*>(B), ldb,
        static_cast<float>(beta),
        static_cast<float*>(C), ldc
    );
}

void dgemm(
    Order order,
    Transposed TransA,
    Transposed TransB,
    int M, int N, int K,
    double alpha,
    const void* A, int lda,
    const void* B, int ldb,
    double beta,
    void* C, int ldc
) {
    cblas_dgemm(
        static_cast<CBLAS_ORDER>(order), 
        static_cast<CBLAS_TRANSPOSE>(TransA), static_cast<CBLAS_TRANSPOSE>(TransB), 
        M, N, K,
        alpha,
        static_cast<const double*>(A), lda,
        static_cast<const double*>(B), ldb,
        beta,
        static_cast<double*>(C), ldc);
}

} // namespace openblas::gemm