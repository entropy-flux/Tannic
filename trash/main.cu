#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512

#define CHECK_CUDA(call)                                                   \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CHECK_CUBLAS(call)                                                \
    do {                                                                 \
        cublasStatus_t status = call;                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                           \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n", __FILE__,\
                    __LINE__, status);                                    \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main() {
    // Select GPU 0 explicitly
    CHECK_CUDA(cudaSetDevice(0));

    size_t size = N * N * sizeof(float);
    float alpha = 1.0f, beta = 0.0f;

    // Host allocations
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_A, N,
                             d_B, N,
                             &beta,
                             d_C, N));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print first 5 results
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}