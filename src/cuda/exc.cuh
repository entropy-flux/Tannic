#pragma once
#include <iostream> 
#include <stdexcept>
#include <cuda_runtime.h>

namespace cuda {

inline void checkError(cudaError_t err, const char* file, int line, const char* expr) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) in '%s'\n",
                file, line, err, cudaGetErrorString(err), expr);
        exit(EXIT_FAILURE);
    }
} 

#define CUDA_CHECK(call) checkError((call), __FILE__, __LINE__, #call)

} // namespace cuda