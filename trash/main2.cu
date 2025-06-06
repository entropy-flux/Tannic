#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}


int main() {
    const int N = 1 << 16; // 65536
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size); // Host buffer.
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); // Cuda buffer!.
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
 

    // Begin capturing
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    vectorAdd<<<(N + 255) / 256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    cudaStreamEndCapture(stream, &graph);

    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Launch the graph
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Validate
    for (int i = 0; i < 10; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream);

    return 0;
}
