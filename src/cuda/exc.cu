#include <cuda_runtime.h>
#include <iostream> 
#include <stdexcept>

namespace cuda {

void checkError(cudaError_t err, const char* file, int line, const char* expr) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) in '%s'\n",
                file, line, err, cudaGetErrorString(err), expr);
        exit(EXIT_FAILURE);
    }
}
 
} // namespace cuda