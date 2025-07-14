#include "cuda/cuda.cuh"

void cuda::checkError(cudaError_t status, const char* message, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream error;
        error << "CUDA Error at " << file << ":" << line << "\n"
              << "  Code: " << static_cast<int>(status) << " (" << cudaGetErrorName(status) << ")\n"
              << "  Message: " << cudaGetErrorString(status);
        if (message && *message)
            error << "\n  Context: " << message;
        throw std::runtime_error(error.str());
    }
}